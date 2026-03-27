import torch
import os
import numpy as np
from logzero import logger

from video_qa.base import BaseVQA, work


class ReKVOfflineVQA(BaseVQA):
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
        
        # Check cache first
        cache_dir = '/mnt/ssd1/mwnoh/qaego4d/cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{video_id}_fps{self.sample_fps}.npy')
        
        if os.path.exists(cache_path):
            logger.debug(f'Loading cached video from {cache_path}')
            video = np.load(cache_path)
        else:
            logger.debug(f'Cache not found, loading video from disk')
            video_path = f'/mnt/ssd1/mwnoh/qaego4d/videos/{video_id}.mp4'
            video = self.load_video(video_path)
            # Save to cache
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

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            
            # QA
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
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']

            self.save_result_to_csv(result_dict)


if __name__ == "__main__":
    work(ReKVOfflineVQA)
