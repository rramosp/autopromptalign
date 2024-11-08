import pandas as pd
from vertexai.generative_models import HarmBlockThreshold, HarmCategory
import vertexai
import asyncio
from vertexai.generative_models import GenerativeModel
import loguru
import numpy as np
import re
import os
from progressbar import progressbar as pbar

log = loguru.logger

extract_answers = lambda s: [i.split('--QUESTION--')[0].strip() for i in s.split('--ANSWER--')][1:]

class TargetModelResponse:

    def __init__(self, system_prompt, generation_model_response=None, score = None):
        self.score = score
        self.system_prompt = system_prompt
        self.generation_model_response = generation_model_response

    def set_evaluation_responses(self, responses, scores):
        self.evaluation_responses = responses
        self.evaluation_scores = scores
        self.score = np.mean(scores)

    def set_response(self, response):
      self.response = response
      self.answers = extract_answers(response)
      return self

    def get_example_string(self, source_model_response):
      """
      builds a string with the system_prompt, the overall mean and the answer getting the minimum score
      """
      ei = np.argmin(self.evaluation_scores)

      s = f'''          
          <SYSTEM_PROMPT>
            <PROMPT_TEXT>{self.system_prompt}</PROMPT_TEXT>
            <OVERALL_SCORE>{self.score}</OVERALL_SCORE>
            <EXAMPLE>
              <LLM_SOURCE_ANSWER>{source_model_response.answers[ei]}</LLM_SOURCE_ANSWER>
              <LLM_TARGET_ANSWER>{self.answers[ei]}</LLM_TARGET_ANSWER>
              <SCORE>{self.evaluation_scores[ei]}</SCORE>
            </EXAMPLE>
          </SYSTEM_PROMPT>
      '''
      return s

    def show_any_question(self, task_questions, source_model_response):

        i = np.random.randint(len(self.answers))

        s = f"""
### Question

{task_questions[i]}

### Source model answer

{source_model_response.answers[i]}

### Target model system prompt
{self.system_prompt}

### Target model answer
{self.answers[i]}
        """
        return display(Markdown(s)) 

class SourceModelResponse:
    def __init__(self, response):
        self.response = response
        self.answers = extract_answers(response)

class LLMAlignment:

    def __init__(self, models_spec):

        self.models_spec = models_spec

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.sampling_system_prompt = ''
        with open('autopromptalign/data/sampling_question.txt') as f:
            self.sampling_question = f.read()

        with open('autopromptalign/data/target_model_task_question_batch.txt') as f:
            self.target_model_task_question_batch = f.read()

        self.source_model_task_system_prompt_batch = ''
        with open('autopromptalign/data/source_model_task_question_batch.txt') as f:
            self.source_model_task_question_batch = f.read()

        self.generation_system_prompt = ''
        with open('autopromptalign/data/generation_question.txt') as f:
            self.generation_question = f.read()

        self.evaluation_system_prompt = ''
        with open('autopromptalign/data/evaluation_question.txt') as f:
            self.evaluation_question = f.read()

        self.target_model_response_history = []

    def sample_task_questions(self, topic_description, number_of_questions=10):
        self.topic_description = topic_description
        q = self.sampling_question.format(number_of_questions=number_of_questions,
                                          topic_description = topic_description)

        m = GenerativeModel(
            self.models_spec['sampling']['name'],
            generation_config=self.models_spec['sampling']['config'],
            safety_settings=self.safety_settings,
            system_instruction=self.sampling_system_prompt
        )

        response = m.generate_content(q,stream=False)

        self.task_questions = response.text
        self.task_questions = [i.strip() for i in self.task_questions.split('--QUESTION--') if len(i.strip())>0]
        log.info(f"sampled {len(self.task_questions)} questions on '{topic_description}'")

    def source_model_generate_answers_batch(self):
        questions = "- "+"\n- ".join(self.task_questions)
        qs = self.source_model_task_question_batch.format(questions=questions)

        m = GenerativeModel(
            self.models_spec['source']['name'],
            generation_config=self.models_spec['source']['config'],
            safety_settings=self.safety_settings,
            system_instruction=self.source_model_task_system_prompt_batch
        )

        response = m.generate_content(qs,stream=False)
        r = SourceModelResponse(response = response.text)
        log.info(f"source model {self.models_spec['source']['name']} batch answered {len(r.answers)} questions on '{self.topic_description}'")
        return r

    def target_model_generate_answers_batch(self, target_model_response: TargetModelResponse = None):
        questions = "- "+"\n- ".join(self.task_questions)
        qs = self.target_model_task_question_batch.format(questions=questions)

        m = GenerativeModel(
            self.models_spec['target']['name'],
            generation_config=self.models_spec['target']['config'],
            safety_settings=self.safety_settings,
            system_instruction=target_model_response.system_prompt
        )

        response = m.generate_content(qs,stream=False)
        target_model_response.set_response(response.text)

        log.info(f"target model {self.models_spec['target']['name']} batch answered {len(target_model_response.answers)} questions on '{self.topic_description}'")
        return target_model_response

    def generate_system_prompt_for_target_model(self):

        example_questions = np.random.permutation(self.task_questions)[:5]
        example_questions = "- "+"\n- ".join(example_questions)

        sorted_prompts = [self.target_model_response_history[i] for i in np.argsort([p.score for p in self.target_model_response_history])[::-1][:10]]

        prompt_scores = "\n".join([p.get_example_string(self.source_model_response) for p in sorted_prompts])

        p = self.generation_question.format(prompt_scores = prompt_scores,
                                            source_model_name = self.models_spec['source']['name'],
                                            target_model_name = self.models_spec['target']['name'],
                                            topic_description = self.topic_description,
                                            example_questions = example_questions)

        g = GenerativeModel(
            self.models_spec['generation']['name'],
            generation_config=self.models_spec['generation']['config'],
            safety_settings=self.safety_settings,
            system_instruction=self.generation_system_prompt
        )

        r = g.generate_content(p,stream=False)

        prompt_extract = re.search('\[\[\[\[(.*)\]\]\]\]', r.text, re.IGNORECASE)

        if prompt_extract:
            self.new_prompt = prompt_extract.group(1)

        else:
            # sometimes the LLM does not follow the instructions exactly
            prompt_extract = re.search('\[\[\[(.*)\]\]\]', r.text, re.IGNORECASE)

            if prompt_extract:
                self.new_prompt = prompt_extract.group(1)
            else:
              self.generation_model_question_instance = p
              self.generation_model_response = r
              raise ValueError('no prompt found in generation model response. see self.generation_model_response and self.generation_model_question_instance')

        self.new_prompt = self.new_prompt.replace('[','').replace(']','')

        return TargetModelResponse(system_prompt = self.new_prompt, generation_model_response = r.text)

    def evaluate_target_response(self, target_respose: TargetModelResponse):
        eval_responses = []
        eval_scores = []

        log.info(f'evaluating similarity of {len(self.task_questions)} questions between source and target model')
        for i in pbar(range(len(self.task_questions))):

            q = self.task_questions[i]
            a_source = self.source_model_response.answers[i]
            a_target = target_respose.answers[i]

            em = GenerativeModel(
                self.models_spec['evaluation']['name'],
                generation_config=self.models_spec['evaluation']['config'],
                safety_settings=self.safety_settings,
                system_instruction=''
            )

            eval_question = self.evaluation_question.format(prompt=q,
                                            baseline_model_response=a_source,
                                            response = a_target)

            r = em.generate_content(eval_question,stream=False)
            eval_responses.append(r.text)
            prompt_extract = re.search('\[\[(.*)\]\]', r.text, re.IGNORECASE)

            eval_scores.append(float(prompt_extract.group(1)))

        target_respose.set_evaluation_responses(eval_responses, eval_scores)

    def alignment_init(self):

        log.info(f"sending the {len(self.task_questions)} questions on '{self.topic_description}' to source model {self.models_spec['source']['name']}")
        self.source_model_response = self.source_model_generate_answers_batch()

        p = TargetModelResponse(system_prompt = f"generate a new prompt for {self.models_spec['target']['name']} to answer questions about '{self.topic_description}' ",
                                generation_model_response= None)
        log.info(f"sending the {len(self.task_questions)} questions on '{self.topic_description}' to target model {self.models_spec['target']['name']}")
        p = self.target_model_generate_answers_batch(target_model_response = p)

        self.evaluate_target_response(p)
        self.target_model_response_history.append(p)


    def alignment_iteration(self):
        log.info('generating new prompt')
        p = self.generate_system_prompt_for_target_model()
        log.info(f"prompt generated '{p.system_prompt}'")
        log.info(f"sending the {len(self.task_questions)} questions on '{self.topic_description}' to target model {self.models_spec['target']['name']}")
        p = self.target_model_generate_answers_batch(target_model_response = p)

        self.evaluate_target_response(p)
        self.target_model_response_history.append(p)


    def history_summary(self):
        return pd.DataFrame([{'system_prompt': p.system_prompt, 'score': p.score} for p in self.target_model_response_history])
    
