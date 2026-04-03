import warnings
import random
import json
import os
import math
import argparse

import pandas as pd
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

from model import llava_onevision_rekv, video_llava_rekv, longva_rekv
from model.profiling import configure_profiling, get_profiler


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': '/mnt/models/llava_ov_7b-hf',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_class': VideoLlavaForConditionalGeneration,
        'processor_class': VideoLlavaProcessor,
        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
    },
    'longva_7b': {
        'load_func': longva_rekv.load_model,
        'model_path': 'model_zoo/LongVA-7B',
    },
}


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1, encode_chunk_size=8,
                 n_local=15000, kv_repr="mean", q_repr="mean",
                 q_token_agg="topk", q_topk_ratio=0.3,
                 k_token_agg="max", k_topk_ratio=0.3,
                 head_specific_retrieval=False, use_video_cache=True,
                 retrieval_fusion="none", fusion_mean_topk=None, fusion_token_topk=None,
                 rerank_candidate_topk=None,
                 profile_video_ids=None, retrieve_sizes=None, rerank_candidate_topks=None) -> None:
        
        self.sample_fps = sample_fps
        self.use_video_cache = use_video_cache

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.retrieve_sizes = list(dict.fromkeys(retrieve_sizes or [retrieve_size]))
        self.chunk_size = chunk_size
        self.encode_chunk_size = encode_chunk_size
        self.n_local = n_local
        self.kv_repr = kv_repr
        self.q_repr = q_repr
        self.q_token_agg = q_token_agg
        self.q_topk_ratio = q_topk_ratio
        self.k_token_agg = k_token_agg
        self.k_topk_ratio = k_topk_ratio
        self.head_specific_retrieval = head_specific_retrieval
        self.retrieval_fusion = retrieval_fusion
        self.rerank_candidate_topk = rerank_candidate_topk
        if self.retrieval_fusion == "rerank" and self.rerank_candidate_topk is None:
            self.rerank_candidate_topk = max(retrieve_size, retrieve_size * 4)
        self.rerank_candidate_topks = list(dict.fromkeys(
            rerank_candidate_topks or [self.rerank_candidate_topk]
        ))
        if fusion_mean_topk is None and fusion_token_topk is None:
            fusion_mean_topk = retrieve_size // 2
            fusion_token_topk = retrieve_size - fusion_mean_topk
        elif fusion_mean_topk is None:
            fusion_mean_topk = retrieve_size - fusion_token_topk
        elif fusion_token_topk is None:
            fusion_token_topk = retrieve_size - fusion_mean_topk
        self.fusion_mean_topk = fusion_mean_topk
        self.fusion_token_topk = fusion_token_topk

        if self.retrieval_fusion != "none":
            assert self.fusion_mean_topk + self.fusion_token_topk == self.retrieve_size, (
                f'fusion_mean_topk + fusion_token_topk must equal retrieve_size: '
                f'{self.fusion_mean_topk} + {self.fusion_token_topk} != {self.retrieve_size}'
            )

        self.retrieval_tag = self._build_retrieval_tag()

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {}
        for size in self.retrieve_sizes:
            if self.retrieval_fusion == "rerank":
                for candidate_topk in self.rerank_candidate_topks:
                    self.record[(size, self.chunk_size, candidate_topk)] = []
            else:
                self.record[(size, self.chunk_size, self.rerank_candidate_topk)] = []
        self.completed_questions = self._load_completed_questions()
        self.profile_video_ids = set(profile_video_ids or [])
        profiler = get_profiler()
        if profiler.is_enabled():
            profiler.update_metadata(
                save_dir=self.save_dir,
                retrieval_tag=self.retrieval_tag,
                retrieve_size=self.retrieve_size,
                chunk_size=self.chunk_size,
                n_local=self.n_local,
                kv_repr=self.kv_repr,
                q_repr=self.q_repr,
                retrieval_fusion=self.retrieval_fusion,
                head_specific_retrieval=self.head_specific_retrieval,
                rerank_candidate_topk=self.rerank_candidate_topk,
            )
            if not self.profile_video_ids:
                profiler.configure(output_path=self.get_profile_json_path())

    def _build_retrieval_tag(self):
        if self.retrieval_fusion != "none":
            if self.retrieval_fusion == "rerank":
                return (
                    f'head_specific_{self.head_specific_retrieval}-fusion_{self.retrieval_fusion}'
                    f'-cand_topk_{self.rerank_candidate_topk}'
                    f'-token_qagg_{self.q_token_agg}-qtopk_{self.q_topk_ratio}'
                    f'-kagg_{self.k_token_agg}-ktopk_{self.k_topk_ratio}'
                )
            return (
                f'head_specific_{self.head_specific_retrieval}-fusion_{self.retrieval_fusion}'
                f'-mean_topk_{self.fusion_mean_topk}-token_topk_{self.fusion_token_topk}'
                f'-token_qagg_{self.q_token_agg}-qtopk_{self.q_topk_ratio}'
                f'-kagg_{self.k_token_agg}-ktopk_{self.k_topk_ratio}'
            )

        if self.kv_repr == "mean":
            if self.q_repr == "mean":
                return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}'
            if self.q_token_agg == "mean":
                return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-qagg_{self.q_token_agg}'
            return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-qagg_{self.q_token_agg}-qtopk_{self.q_topk_ratio}'

        if self.k_token_agg == "max":
            if self.q_repr == "mean":
                return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-kagg_{self.k_token_agg}'
            if self.q_token_agg == "mean":
                return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-qagg_{self.q_token_agg}-kagg_{self.k_token_agg}'
            return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-qagg_{self.q_token_agg}-qtopk_{self.q_topk_ratio}-kagg_{self.k_token_agg}'

        if self.q_repr == "mean":
            return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-kagg_{self.k_token_agg}-ktopk_{self.k_topk_ratio}'
        if self.q_token_agg == "mean":
            return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-qagg_{self.q_token_agg}-kagg_{self.k_token_agg}-ktopk_{self.k_topk_ratio}'
        return f'head_specific_{self.head_specific_retrieval}-kv_repr_{self.kv_repr.replace("_", "-")}-q_repr_{self.q_repr}-qagg_{self.q_token_agg}-qtopk_{self.q_topk_ratio}-kagg_{self.k_token_agg}-ktopk_{self.k_topk_ratio}'

    def _current_record_key(self):
        return (self.retrieve_size, self.chunk_size, self.rerank_candidate_topk)

    def set_retrieval_config(self, retrieve_size=None, rerank_candidate_topk=None):
        if retrieve_size is not None:
            self.retrieve_size = retrieve_size
        if rerank_candidate_topk is not None:
            self.rerank_candidate_topk = rerank_candidate_topk
        self.retrieval_tag = self._build_retrieval_tag()
        self.record.setdefault(self._current_record_key(), [])
        self.completed_questions = self._load_completed_questions()
        profiler = get_profiler()
        if profiler.is_enabled():
            profiler.update_metadata(
                retrieval_tag=self.retrieval_tag,
                retrieve_size=self.retrieve_size,
                rerank_candidate_topk=self.rerank_candidate_topk,
            )
        if hasattr(self.qa_model, "set_retrieve_size"):
            self.qa_model.set_retrieve_size(self.retrieve_size)
        if hasattr(self.qa_model, "set_rerank_candidate_topk"):
            self.qa_model.set_rerank_candidate_topk(self.rerank_candidate_topk)

    def set_retrieve_size(self, retrieve_size):
        self.retrieve_size = retrieve_size
        self.set_retrieval_config(retrieve_size=retrieve_size)

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path):
        resolved_video_path = os.path.abspath(os.path.expanduser(str(video_path)))
        if resolved_video_path.endswith('.npy'):
            video = np.load(resolved_video_path)
            logger.debug(f'loaded numpy video from {resolved_video_path} with shape {video.shape}')
            return video

        if not os.path.exists(resolved_video_path):
            raise FileNotFoundError(
                f'Video file not found: {resolved_video_path} '
                f'(original={video_path}, cwd={os.getcwd()})'
            )

        if not os.path.isfile(resolved_video_path):
            raise RuntimeError(
                f'Video path is not a file: {resolved_video_path} '
                f'(original={video_path})'
            )

        try:
            vr = VideoReader(resolved_video_path, ctx=cpu(0), num_threads=1)
        except Exception as exc:
            file_size = os.path.getsize(resolved_video_path)
            raise RuntimeError(
                f'Failed to open video with decord: {resolved_video_path} '
                f'(original={video_path}, size_bytes={file_size}, cwd={os.getcwd()})'
            ) from exc

        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape} from {resolved_video_path}')
        return video
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def save_result_to_csv(self, result_dict):
        """Save a single result to CSV file by appending"""
        csv_path = self.get_result_csv_path()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        df = pd.DataFrame([result_dict])
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        self.mark_question_completed(result_dict)

    def get_result_csv_path(self):
        return f'{self.save_dir}/{self.retrieval_tag}/n{self.n_local}/{self.retrieval_tag}_rs{self.retrieve_size}_cs{self.chunk_size}_n{self.n_local}_accuracy.csv'

    def get_profile_json_path(self):
        return f'{self.save_dir}/{self.retrieval_tag}/n{self.n_local}/{self.retrieval_tag}_rs{self.retrieve_size}_cs{self.chunk_size}_n{self.n_local}_profile.json'

    def get_video_profile_json_path(self, video_id):
        return (
            f'{self.save_dir}/{self.retrieval_tag}/n{self.n_local}/profiles/'
            f'{self.retrieval_tag}_rs{self.retrieve_size}_cs{self.chunk_size}_n{self.n_local}_{video_id}_profile.json'
        )

    def get_profile_selected_json_path(self):
        return (
            f'{self.save_dir}/{self.retrieval_tag}/n{self.n_local}/profiles/'
            f'{self.retrieval_tag}_rs{self.retrieve_size}_cs{self.chunk_size}_n{self.n_local}_selected_videos_profile.json'
        )

    def _result_key_columns(self, df):
        if 'question_idx' in df.columns:
            return 'question_idx'
        if 'question' in df.columns:
            return 'question'
        return None

    def _load_completed_questions(self):
        csv_path = self.get_result_csv_path()
        if not os.path.exists(csv_path):
            return {}

        df = pd.read_csv(csv_path)
        if df.empty or 'video_id' not in df.columns:
            return {}

        key_column = self._result_key_columns(df)
        if key_column is None:
            return {}

        completed = {}
        for _, row in df[['video_id', key_column]].dropna().iterrows():
            video_id = str(row['video_id'])
            question_key = str(row[key_column])
            completed.setdefault(video_id, set()).add(question_key)
        logger.info(f'Loaded {sum(len(v) for v in completed.values())} completed questions from {csv_path}')
        return completed

    def get_completed_questions(self, video_id):
        return self.completed_questions.get(str(video_id), set())

    def is_question_completed(self, video_id, question_key):
        if question_key is None:
            return False
        return str(question_key) in self.get_completed_questions(video_id)

    def mark_question_completed(self, result_dict):
        video_id = result_dict.get('video_id')
        question_key = result_dict.get('question_idx', result_dict.get('question'))
        if video_id is None or question_key is None:
            return
        self.completed_questions.setdefault(str(video_id), set()).add(str(question_key))

    def analyze(self, debug=False):
        profiler = get_profiler()
        selected_video_summaries = []
        try:
            video_annos = self.anno[:1] if debug else self.anno
            if self.profile_video_ids:
                video_annos = [
                    video_sample for video_sample in video_annos
                    if str(video_sample["video_id"]) in self.profile_video_ids
                ]
                logger.info(
                    f'Filtered videos for profiling/analysis: '
                    f'{len(video_annos)} selected out of {len(self.anno[:1] if debug else self.anno)}'
                )
            for video_sample in tqdm(video_annos):
                video_id = str(video_sample["video_id"])
                logger.debug(f'video_id: {video_id}')
                should_profile = profiler.is_enabled() and (
                    not self.profile_video_ids or video_id in self.profile_video_ids
                )

                if profiler.is_enabled() and self.profile_video_ids:
                    profiler.configure(
                        enabled=should_profile,
                        output_path=self.get_video_profile_json_path(video_id) if should_profile else None,
                        reset=should_profile,
                    )
                    if should_profile:
                        profiler.update_metadata(
                            video_id=video_id,
                            video_duration=video_sample.get("duration"),
                            num_questions=len(video_sample.get("conversations", [])),
                        )

                self.analyze_a_video(video_sample)

                if should_profile:
                    summary = profiler.summary()
                    profiler.dump()
                    selected_video_summaries.append(summary)

            if not any(self.record.values()):
                logger.info('No new results were generated in this run.')
                return

            dfs = []
            for (retrieve_size, chunk_size, rerank_candidate_topk), dict_list in self.record.items():
                df = pd.DataFrame(dict_list)
                df['retrieve_size'] = retrieve_size
                df['chunk_size'] = chunk_size
                df['rerank_candidate_topk'] = rerank_candidate_topk
                dfs.append(df)
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)
        finally:
            if profiler.is_enabled():
                if self.profile_video_ids:
                    if selected_video_summaries:
                        output_path = self.get_profile_selected_json_path()
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(
                                {
                                    'enabled': True,
                                    'metadata': {
                                        'save_dir': self.save_dir,
                                        'retrieval_tag': self.retrieval_tag,
                                        'selected_video_ids': sorted(self.profile_video_ids),
                                    },
                                    'videos': selected_video_summaries,
                                },
                                f,
                                indent=2,
                                ensure_ascii=True,
                            )
                else:
                    profiler.dump()


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_sizes", type=str, default="")
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--encode_chunk_size", type=int, default=8)
    parser.add_argument("--kv_repr", type=str, default="mean")
    parser.add_argument("--q_repr", type=str, default="mean")
    parser.add_argument("--q_token_agg", type=str, default="topk", choices=["mean", "topk"])
    parser.add_argument("--q_topk_ratio", type=float, default=0.3)
    parser.add_argument("--k_token_agg", type=str, default="max", choices=["max", "topk"])
    parser.add_argument("--k_topk_ratio", type=float, default=0.3)
    parser.add_argument("--head_specific_retrieval", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--retrieval_fusion", type=str, default="none", choices=["none", "quota", "rerank"])
    parser.add_argument("--fusion_mean_topk", type=int, default=None)
    parser.add_argument("--fusion_token_topk", type=int, default=None)
    parser.add_argument("--rerank_candidate_topk", type=int, default=None)
    parser.add_argument("--rerank_candidate_topks", type=str, default="")
    parser.add_argument("--use_video_cache", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--profile_perf", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--profile_video_ids", type=str, default="")
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)
    configure_profiling(enabled=args.profile_perf, reset=True)
    profile_video_ids = [x.strip() for x in args.profile_video_ids.split(",") if x.strip()]
    retrieve_sizes = [int(x.strip()) for x in args.retrieve_sizes.split(",") if x.strip()]
    if not retrieve_sizes:
        retrieve_sizes = [args.retrieve_size]
    rerank_candidate_topks = [int(x.strip()) for x in args.rerank_candidate_topks.split(",") if x.strip()]
    if not rerank_candidate_topks:
        rerank_candidate_topks = [args.rerank_candidate_topk]
    model_retrieve_size = max(retrieve_sizes)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        n_local=args.n_local,
        topk=model_retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        kv_repr=args.kv_repr,
        q_repr=args.q_repr,
        q_token_agg=args.q_token_agg,
        q_topk_ratio=args.q_topk_ratio,
        k_token_agg=args.k_token_agg,
        k_topk_ratio=args.k_topk_ratio,
        head_specific_retrieval=args.head_specific_retrieval,
        retrieval_fusion=args.retrieval_fusion,
        fusion_mean_topk=args.fusion_mean_topk,
        fusion_token_topk=args.fusion_token_topk,
        rerank_candidate_topk=args.rerank_candidate_topk,
    )

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        retrieve_sizes=retrieve_sizes,
        rerank_candidate_topks=rerank_candidate_topks,
        chunk_size=args.retrieve_chunk_size,
        encode_chunk_size=args.encode_chunk_size,
        n_local=args.n_local,
        kv_repr=args.kv_repr,
        q_repr=args.q_repr,
        q_token_agg=args.q_token_agg,
        q_topk_ratio=args.q_topk_ratio,
        k_token_agg=args.k_token_agg,
        k_topk_ratio=args.k_topk_ratio,
        head_specific_retrieval=args.head_specific_retrieval,
        retrieval_fusion=args.retrieval_fusion,
        fusion_mean_topk=args.fusion_mean_topk,
        fusion_token_topk=args.fusion_token_topk,
        rerank_candidate_topk=args.rerank_candidate_topk,
        use_video_cache=args.use_video_cache,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
        profile_video_ids=profile_video_ids,
    )

    retrieve_analyzer.analyze(debug=args.debug)
