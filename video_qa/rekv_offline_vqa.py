import torch
import os
import re
import numpy as np
from pathlib import Path
from logzero import logger

from video_qa.base import BaseVQA, work


class ReKVOfflineVQA(BaseVQA):
    LVBENCH_CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    def __init__(self, *args, **kwargs):
        anno = kwargs.get('anno')
        if anno and 'qa' in anno[0]:
            kwargs['anno'] = [self.normalize_lvbench_sample(item) for item in anno]
        super().__init__(*args, **kwargs)

    def parse_lvbench_question(self, raw_question):
        question_lines = []
        choices = []
        for line in raw_question.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r'^\(([A-H])\)\s*(.*)$', stripped)
            if match:
                choices.append(match.group(2).strip())
            else:
                question_lines.append(stripped)
        question = " ".join(question_lines).strip()
        return question, choices

    def normalize_lvbench_sample(self, item):
        conversations = []
        for qa in item.get('qa', []):
            question, choices = self.parse_lvbench_question(qa['question'])
            answer_letter = qa.get('answer')
            answer = None
            if answer_letter in self.LVBENCH_CHOICE_LETTERS and choices:
                answer_idx = self.LVBENCH_CHOICE_LETTERS.index(answer_letter)
                if answer_idx < len(choices):
                    answer = choices[answer_idx]
            conversations.append({
                'question_idx': qa.get('uid'),
                'question': question,
                'choices': choices,
                'answer': answer,
                'question_type': ", ".join(qa.get('question_type', [])),
                'time_reference': qa.get('time_reference'),
            })

        video_info = item.get('video_info', {})
        duration_minutes = video_info.get('duration_minutes')
        return {
            'video_id': item['key'],
            'video_path': item['downloaded_video_path'],
            'dataset': 'lvbench',
            'type': item.get('type'),
            'duration': duration_minutes * 60 if duration_minutes is not None else None,
            'conversations': conversations,
        }

    def resolve_video_path(self, video_sample):
        video_path = video_sample.get('video_path')
        if video_path:
            if os.path.isabs(video_path):
                return video_path
            project_root = Path(__file__).resolve().parent.parent
            return str(project_root / video_path)
        return f'/mnt/ssd1/mwnoh/qaego4d/videos/{video_sample["video_id"]}.mp4'

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_id = video_sample['video_id']
        dataset_name = video_sample.get('dataset', 'qaego4d')
        video_path = self.resolve_video_path(video_sample)
        logger.debug(f'Resolved video path for {video_id}: {video_path}')
        
        # Check cache first
        cache_dir = os.environ.get('REKV_VIDEO_CACHE_DIR', f'/tmp/rekv_video_cache/{dataset_name}')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{video_id}_fps{self.sample_fps}.npy')
        
        if self.use_video_cache and os.path.exists(cache_path):
            logger.debug(f'Loading cached video from {cache_path}')
            video = np.load(cache_path)
        else:
            if self.use_video_cache:
                logger.debug(f'Cache not found, loading video from disk')
            else:
                logger.debug('Video cache disabled, loading video from disk')
            video = self.load_video(video_path)
            if self.use_video_cache:
                logger.debug(f'Saving video to cache: {cache_path}')
                np.save(cache_path, video)
        # Get number of frames
        num_frames = video.shape[0]
        logger.debug(f'Number of frames: {num_frames}')

        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor, self.encode_chunk_size)

        if self.retrieval_fusion == "rerank":
            retrieval_configs = [
                (retrieve_size, candidate_topk)
                for candidate_topk in self.rerank_candidate_topks
                for retrieve_size in self.retrieve_sizes
            ]
        else:
            retrieval_configs = [(retrieve_size, self.rerank_candidate_topk) for retrieve_size in self.retrieve_sizes]

        for retrieve_size, rerank_candidate_topk in retrieval_configs:
            self.set_retrieval_config(
                retrieve_size=retrieve_size,
                rerank_candidate_topk=rerank_candidate_topk,
            )
            logger.info(
                f'Running video {video_id} with retrieve_size={retrieve_size}, '
                f'rerank_candidate_topk={rerank_candidate_topk}'
            )

            for sample in video_sample['conversations']:
                logger.debug(f'sample: {sample}')
                question = sample['question']
                answer = sample['answer']

                if 'choices' in sample:  # CloseQA
                    choices = sample['choices']
                    if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                        answer = choices[0]
                    correct_choice = self.choice_letters[choices.index(answer)]
                    qa_results = self.video_close_qa(question, choices, correct_choice)
                    result_dict = {
                        'video_id': video_sample['video_id'],
                        'question': question,
                        'choices': choices,
                        'answer': answer,
                        'correct_choice': correct_choice,
                        'pred_answer': qa_results['pred_answer'],
                        'pred_choice': qa_results['pred_choice'],
                        'qa_acc': qa_results['acc'] * 100,
                        'retrieve_size': self.retrieve_size,
                        'chunk_size': self.chunk_size,
                    }
                else:  # OpenQA
                    qa_results = self.video_open_qa(question)
                    result_dict = {
                        'video_id': video_sample['video_id'],
                        'question': question,
                        'answer': answer,
                        'pred_answer': qa_results['pred_answer'],
                        'retrieve_size': self.retrieve_size,
                        'chunk_size': self.chunk_size,
                    }

                if 'question_type' in sample:
                    result_dict['task'] = sample['question_type']

                self.record[self._current_record_key()].append(result_dict)
                self.save_result_to_csv(result_dict)


if __name__ == "__main__":
    work(ReKVOfflineVQA)
