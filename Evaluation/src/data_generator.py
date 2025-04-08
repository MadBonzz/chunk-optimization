from openai import OpenAI
from .prompts import context_evaluation as ce
from .prompts import qa_generation as qa
from .prompts import qa_eval as qae
import json
import re

class DataGenerator:
    def __init__(self, generator_model_id, discriminator_model_id, base_url):
        self.generator_model = generator_model_id
        self.discriminator_model = discriminator_model_id
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key="lm-studio")

    def context_preprocess(self, chunks, output_dir):
        file = open(output_dir, 'a')
        file.write('[\n')
        for i in range(len(chunks)):
            eval_prompt = ce.evaluation_prompt
            evaluation = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI context evaluator. You are tasked to evaluate if the provided chunk has enough context 
                                    for creating a factual question."""
                    },
                    {
                        "role": "user",
                        "content": eval_prompt.format(
                            rules = ce.rules,
                            guidelines = ce.guidelines,
                            examples = ce.examples,
                            context = chunks[i]
                        )
                    }
                ],
                tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Context-Evaluator",
                        "parameters": {
                        "type": "object",
                        "title": "Context-Evaluator",
                        "properties": {
                            "reasoning": {
                            "title": "Reasoning",
                            "type": "string",
                            "description": "Why or Why not the context has enough information"
                            },
                            "rating": {
                            "title": "Rating",
                            "type": "float",
                            "description": "Rating from 0 to 1 about context richness of the chunk. 1 is the highest, 0 is the lowest"
                            }
                        },
                        "required": [
                            "reasoning",
                            "rating"
                        ]
                        }
                    }
                }  
                ],
                tool_choice={"type": "function", "function": {"name": "Context-Evaluator"}},
                temperature=0.5,
            )
            try:
                output = eval(evaluation.choices[0].message.tool_calls[0].function.arguments)
                if "rating" not in output.keys():
                    print(i)
                    print(output)
                if float(output["rating"]) > 0.5:
                    output["index"] = i
                    output["text"] = chunks[i]
                    output = json.dumps(output, indent=4)
                    file.write(output)
                    if i != (len(chunks) - 1):
                        file.write(',\n')
            except:
                print(evaluation)
                print(output)
        file.write('\n]')
        file.close()

    def generate_qa(self, chunks, output_dir):
        pattern = re.compile(
            r'"question"\s*:\s*"(?P<question>(?:\\.|[^"\\])*)"[\s\S]*?"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)"',
            re.DOTALL
        )
        file = open(output_dir, 'a')
        file.write('[\n')
        for i in range(len(chunks)):
            qa_prompt = qa.question_answer_prompt
            generation = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI question and answer generator. Given the context, create a question for it,
                        and also provide the corresponding answer. Follow the user prompt accurately."""
                    },
                    {
                        "role": "user",
                        "content": qa_prompt.format(
                            context = chunks[i]
                        )
                    }
                ],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "QuestionAnswerGenerator",
                            "parameters": {
                                "type": "object",
                                "title": "QuestionAnswerGenerator",
                                "properties": {
                                    "question": {
                                        "title": "Question",
                                        "type": "string",
                                        "description": "The generated fact-based question derived from the context. The question should be open-ended enough to invite a detailed answer, encouraging deeper analysis rather than a simple, direct factual query."
                                    },
                                    "answer": {
                                        "title": "Answer",
                                        "type": "string",
                                        "description": "The detailed answer corresponding to the generated question, directly based on the information provided in the context."
                                    }
                                },
                                "required": [
                                    "question",
                                    "answer"
                                ]
                            }
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "QuestionAnswerGenerator"}},
                temperature=0.7,
            )

            response = generation.choices[0].message.content
            qa_dict = None
            match = pattern.search(response)
            if match:
                qa_dict = match.groupdict()
            if qa_dict:
                qa_dict["context"] = chunks[i]
                output = json.dumps(qa_dict, indent=4)
                file.write(output)
                if i != (len(chunks) - 1):
                    file.write(',\n')
        file.write('\n]')
        file.close()

    def qac_evaluator(self, qac, output_dir):
        pattern = re.compile(
                r'"evaluation"\s*:\s*"(?P<evaluation>(?:\\.|[^"\\])*)"[^}]*'
                r'"relevance_rating"\s*:\s*(?P<relevance_rating>\d+),\s*'
                r'"groundness_rating"\s*:\s*(?P<groundness_rating>\d+),\s*'
                r'"standalone_rating"\s*:\s*(?P<standalone_rating>\d+)',
                re.DOTALL
        )
        qac_eval = qae.question_evaluation_prompt
        file = open(output_dir, 'a')
        file.write('[\n')
        for i in range(len(qac)):
            generation = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI question evaluator. Given the context and the corresponding question,
                        evaluate the question on relevance, groundness and standalone levels of the question."""
                    },
                    {
                        "role": "user",
                        "content": qac_eval.format(
                            context = qac[i]['context'],
                            question = qac[i]['question']
                        )
                    }
                ],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "QuestionEvaluator",
                            "parameters": {
                                "type": "object",
                                "title": "QuestionEvaluator",
                                "properties": {
                                    "evaluation": {
                                        "title": "Evaluation",
                                        "type": "string",
                                        "description": "A detailed evaluation explaining why the question is appropriate or not for the given context based on relevance, groundness, and its ability to stand alone."
                                    },
                                    "relevance_rating": {
                                        "title": "Relevance Rating",
                                        "type": "number",
                                        "description": "A rating from 1 to 5 indicating how relevant the question is to the provided context."
                                    },
                                    "groundness_rating": {
                                        "title": "Groundness Rating",
                                        "type": "number",
                                        "description": "A rating from 1 to 5 indicating how well the context provides adequate information to answer the question."
                                    },
                                    "standalone_rating": {
                                        "title": "Standalone Rating",
                                        "type": "number",
                                        "description": "A rating from 1 to 5 indicating how well the question stands on its own without needing additional context."
                                    }
                                },
                                "required": [
                                    "evaluation",
                                    "relevance_rating",
                                    "groundness_rating",
                                    "standalone_rating"
                                ]
                            }
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "QuestionEvaluator"}},
                temperature=0.5,
            )
            response = generation.choices[0].message.content
            response = str(response)
            match = pattern.search(response)
            if match:
                result = match.groupdict()
                result['relevance_rating'] = int(result['relevance_rating'])
                result['groundness_rating'] = int(result['groundness_rating'])
                result['standalone_rating'] = int(result['standalone_rating'])
                for key in qac[i].keys():
                    result[key] = qac[i][key]
                output = json.dumps(result, indent=4)
                file.write(output)
                if i != (len(qac)):
                    file.write(',\n')
        file.write('\n]')
        file.close()

            
