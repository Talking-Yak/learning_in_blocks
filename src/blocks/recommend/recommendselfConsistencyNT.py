import os
import pandas as pd
import json
import time
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
cwd = os.getcwd()
env_path = os.path.join(cwd, '.env')
load_dotenv(dotenv_path=env_path)

class RecommendSelfConsistencyProcessor:
    def __init__(self):
        # Configure Novita AI API
        novita_key = os.getenv('NOVITA_API_KEY')
        if not novita_key:
            raise ValueError("NOVITA_API_KEY not found in environment variables.")
        
        self.novita_client = OpenAI(
            api_key=novita_key,
            base_url="https://api.novita.ai/openai"
        )
        self.qwen_model_name = "qwen/qwen3-30b-a3b-fp8" 
        self.judge_model = "qwen/qwen3-30b-a3b-fp8"

        # File paths
        self.source_csv_path = "asset/CSVs/dd_processed.csv"
        self.output_csv_path = "asset/CSVs/recommendSelfConsistencyNT_output.csv" 
        
        self.prompt_dir = "asset/recommend/"
        
        # Context Data Paths
        self.grammar_csv_path = "asset/grammar_flow.csv"
        self.vocab_json_path = "asset/vocab.json"
        
        # Evaluation Data Path
        self.eval_csv_path = "asset/CSVs/selfConsistencyNT1.csv"

        # Define output columns
        self.final_columns = [
            'learnerId', 
            'conversationHistoryCleaned', 
            'final_recommendation', 
            'response1', 'response2', 'response3',
            'tokens_response1', 'tokens_response2', 'tokens_response3',
            'qwen_consensus_response'
        ]
        
        # Load logic
        self.load_context()
        self.load_prompts()
        self.df = self.initialize_data()

    def load_context(self):
        # Load Grammar Context
        df_grammar = pd.read_csv(self.grammar_csv_path)
        grammar_list = []
        for _, row in df_grammar.iterrows():
            entry = f"ID {row['skill_no']}: {row['grammar_skills']} - {str(row['skill_description'])[:100]}..."
            grammar_list.append(entry)
        self.grammar_context = "\n".join(grammar_list)
        
        # Load Vocab Context
        with open(self.vocab_json_path, 'r') as f:
            vocab_data = json.load(f)
        vocab_list = []
        for topic, words in vocab_data.items():
            word_str = ", ".join(words)
            entry = f"Topic: {topic}\nWords: {word_str}\n"
            vocab_list.append(entry)
        self.vocab_context = "\n".join(vocab_list)

        # 3. Load Evaluation Context
        if os.path.exists(self.eval_csv_path):
            print(f"Loading Evaluation Context from {self.eval_csv_path}...")
            self.eval_df = pd.read_csv(self.eval_csv_path)[['learnerId', 'conversationHistoryCleaned', 'score', 'feedback']]
        else:
            print(f"Warning: Evaluation file not found: {self.eval_csv_path}")
            self.eval_df = pd.DataFrame(columns=['learnerId', 'conversationHistoryCleaned', 'score', 'feedback'])

    def get_evaluation_context(self, learnerId, transcript):
        row = self.eval_df[(self.eval_df['learnerId'] == learnerId) & (self.eval_df['conversationHistoryCleaned'] == transcript)]
        if not row.empty:
            return str(row.iloc[0]['score']), str(row.iloc[0]['feedback'])
        return "N/A", "N/A"

    def load_prompts(self):
        # We use AgentC as the balanced recommendation prompt
        with open(f"{self.prompt_dir}AgentC_Conversation.txt", 'r') as f:
            self.recommend_prompt_template = f.read()
            
        self.consensus_template = """
You are an expert judge. You are given a student's conversation transcript and 3 different recommendation analyses provided by an AI.
Your task is to analyze these 3 responses and the transcript to arrive at a final consensus recommendation.

TRANSCRIPT:
{transcript}

RESPONSE 1:
{response1}

RESPONSE 2:
{response2}

RESPONSE 3:
{response3}

Please evaluate the consistency among the responses. Choose the most accurate and helpful feedback based on the transcript.
Output the final recommendation in the following JSON format:
{
  "feedback_well": "...",
  "feedback_improve": "...",
  "grammar_skills": [ID1, ID2],
  "vocab_skills": ["Topic1", "Topic2"]
}
Only output the JSON.
"""

    def initialize_data(self):
        if os.path.exists(self.output_csv_path):
            df = pd.read_csv(self.output_csv_path)
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
            return df
        else:
            df = pd.read_csv(self.source_csv_path)
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
            df.to_csv(self.output_csv_path, index=False)
            return df

    def call_qwen(self, prompt, temperature=0.7):
        while True:
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.qwen_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                token_usage = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
                return content, token_usage
            except Exception as e:
                time.sleep(5)

    def call_judge(self, prompt):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant. You must output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.1,
                    response_format={ "type": "json_object" }
                )
                raw_text = response.choices[0].message.content
                raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                token_usage = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
                return json.loads(raw_text), raw_text, token_usage
            except Exception as e:
                time.sleep(2)
        return None, None, 0

    def get_consensus(self, transcript, responses):
        prompt = self.consensus_template.replace("{transcript}", transcript)\
                                               .replace("{response1}", responses[0])\
                                               .replace("{response2}", responses[1])\
                                               .replace("{response3}", responses[2])
        return self.call_judge(prompt)

    def process(self):
        total_rows = len(self.df)
        print(f"Total rows in dataframe: {total_rows}")
        
        for index, row in self.df.iterrows():
            if pd.notna(row['final_recommendation']) and str(row['final_recommendation']).strip() != "":
                 continue

            transcript = row['conversationHistoryCleaned']
            if not isinstance(transcript, str) or not transcript.strip():
                print(f"Skipping Row {index+1}: Empty transcript")
                continue

            print(f"Processing Row {index+1}/{total_rows}")
            
            # Get Evaluation Context
            score_ctx, feedback_ctx = self.get_evaluation_context(row['learnerId'], transcript)
            
            prompt = self.recommend_prompt_template.replace("{transcript}", transcript)\
                                                   .replace("{grammar_context}", self.grammar_context)\
                                                   .replace("{vocab_context}", self.vocab_context)\
                                                   .replace("{score}", score_ctx)\
                                                   .replace("{feedback}", feedback_ctx)
            
            responses = []
            tokens = []
            for i in range(3):
                res, tok = self.call_qwen(prompt, temperature=0.7)
                responses.append(res)
                tokens.append(tok)
                self.df.at[index, f'response{i+1}'] = res
                self.df.at[index, f'tokens_response{i+1}'] = tok
            
            consensus_json, raw_consensus, _ = self.get_consensus(transcript, responses)
            if consensus_json:
                self.df.at[index, 'final_recommendation'] = json.dumps(consensus_json)
                self.df.at[index, 'qwen_consensus_response'] = raw_consensus
            
            self.df.to_csv(self.output_csv_path, index=False)

if __name__ == "__main__":
    processor = RecommendSelfConsistencyProcessor()
    processor.process()
