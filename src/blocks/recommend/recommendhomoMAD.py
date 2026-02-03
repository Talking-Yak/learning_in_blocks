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

class RecommendHomoMADProcessor:
    def __init__(self):
        # Configure Novita AI API
        novita_key = os.getenv('NOVITA_API_KEY')
        if not novita_key:
            raise ValueError("NOVITA_API_KEY not found in environment variables.")
        
        self.novita_client = OpenAI(
            api_key=novita_key,
            base_url="https://api.novita.ai/openai"
        )
        
        # In HomoMAD, all agents use the same model
        self.common_model = "qwen/qwen3-30b-a3b-fp8"
        self.judge_model = "qwen/qwen3-30b-a3b-fp8"

        # File paths
        self.source_csv_path = "asset/CSVs/dd_processed.csv"
        self.output_csv_path = "asset/CSVs/recommendhomoMAD_output.csv" 
        
        self.prompt_dir = "asset/recommend/"
        
        # Context Data Paths
        self.grammar_csv_path = "asset/grammar_flow.csv"
        self.vocab_json_path = "asset/vocab.json"

        # Evaluation Data Path
        self.eval_csv_path = "asset/CSVs/homoMAD_output.csv"

        # Define output columns
        self.final_columns = [
            'learnerId', 
            'conversationHistoryCleaned', 
            'final_recommendation', 
            'rec_a_initial', 'rec_b_initial', 'rec_c_initial',
            'rec_a_final', 'rec_b_final', 'rec_c_final',
            'judge_response',
            'tokens_a_init', 'tokens_b_init', 'tokens_c_init',
            'tokens_a_final', 'tokens_b_final', 'tokens_c_final',
            'tokens_judge'
        ]
        
        # Load logic
        self.load_context()
        self.load_prompts()
        self.df = self.initialize_data()

    def load_context(self):
        # 1. Load Grammar Context
        if not os.path.exists(self.grammar_csv_path):
             raise FileNotFoundError(f"Grammar file not found: {self.grammar_csv_path}")
        
        df_grammar = pd.read_csv(self.grammar_csv_path)
        grammar_list = []
        for _, row in df_grammar.iterrows():
            entry = f"ID {row['skill_no']}: {row['grammar_skills']} - {str(row['skill_description'])[:100]}..."
            grammar_list.append(entry)
        self.grammar_context = "\n".join(grammar_list)
        
        # 2. Load Vocab Context
        if not os.path.exists(self.vocab_json_path):
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_json_path}")
        
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
        with open(f"{self.prompt_dir}AgentA_Grammar.txt", 'r') as f:
            self.agent_a_prompt = f.read()
        with open(f"{self.prompt_dir}AgentB_Vocab.txt", 'r') as f:
            self.agent_b_prompt = f.read()
        with open(f"{self.prompt_dir}AgentC_Conversation.txt", 'r') as f:
            self.agent_c_prompt = f.read()
            
        with open(f"{self.prompt_dir}Critique_Instruction.txt", 'r') as f:
            self.critique_prompt = f.read()
        
        with open(f"{self.prompt_dir}judge.txt", 'r') as f:
            self.judge_prompt = f.read()

    def initialize_data(self):
        if os.path.exists(self.output_csv_path):
            df = pd.read_csv(self.output_csv_path)
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
                elif col in ['final_recommendation', 'rec_a_initial', 'rec_b_initial', 'rec_c_initial', 'rec_a_final', 'rec_b_final', 'rec_c_final', 'judge_response']:
                     df[col] = df[col].astype('object')
            df = df[self.final_columns]
            return df
        else:
            if not os.path.exists(self.source_csv_path):
                raise FileNotFoundError(f"Input CSV not found at {self.source_csv_path}")
            df = pd.read_csv(self.source_csv_path)
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
            df = df[self.final_columns]
            df.to_csv(self.output_csv_path, index=False)
            return df

    def call_novita(self, prompt, temperature=0.7):
        while True:
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.common_model,
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

    def process(self):
        total_rows = len(self.df)
        for index, row in self.df.iterrows():
            if pd.notna(row['final_recommendation']):
                 continue

            transcript = row['conversationHistoryCleaned']
            if not isinstance(transcript, str) or not transcript.strip():
                continue

            print(f"Processing Row {index+1}/{total_rows}")
            
            # --- PHASE 1 ---
            p_a = self.agent_a_prompt.replace("{transcript}", transcript)\
                                     .replace("{grammar_context}", self.grammar_context)\
                                     .replace("{vocab_context}", self.vocab_context)
            res_a_init, tok_a_init = self.call_novita(p_a)
            self.df.at[index, 'rec_a_initial'] = res_a_init
            self.df.at[index, 'tokens_a_init'] = tok_a_init
            
            p_b = self.agent_b_prompt.replace("{transcript}", transcript)\
                                     .replace("{grammar_context}", self.grammar_context)\
                                     .replace("{vocab_context}", self.vocab_context)
            res_b_init, tok_b_init = self.call_novita(p_b)
            self.df.at[index, 'rec_b_initial'] = res_b_init
            self.df.at[index, 'tokens_b_init'] = tok_b_init
            
            p_c = self.agent_c_prompt.replace("{transcript}", transcript)\
                                     .replace("{grammar_context}", self.grammar_context)\
                                     .replace("{vocab_context}", self.vocab_context)
            res_c_init, tok_c_init = self.call_novita(p_c)
            self.df.at[index, 'rec_c_initial'] = res_c_init
            self.df.at[index, 'tokens_c_init'] = tok_c_init
            
            # Get Evaluation Context
            score_ctx, feedback_ctx = self.get_evaluation_context(row['learnerId'], transcript)

            # --- PHASE 2 ---
            peers_for_a = f"--- Peer B (Vocab) ---\n{res_b_init}\n\n--- Peer C (Conversation) ---\n{res_c_init}"
            p_reflect_a = self.critique_prompt.replace("{agent_role}", "Agent A (Grammar Expert)")\
                                              .replace("{my_previous_response}", res_a_init)\
                                              .replace("{peer_responses}", peers_for_a)\
                                              .replace("{transcript}", transcript)\
                                              .replace("{score}", score_ctx)\
                                              .replace("{feedback}", feedback_ctx)

            res_a_final, tok_a_final = self.call_novita(p_reflect_a)
            self.df.at[index, 'rec_a_final'] = res_a_final
            self.df.at[index, 'tokens_a_final'] = tok_a_final
            
            peers_for_b = f"--- Peer A (Grammar) ---\n{res_a_init}\n\n--- Peer C (Conversation) ---\n{res_c_init}"
            p_reflect_b = self.critique_prompt.replace("{agent_role}", "Agent B (Vocabulary Expert)")\
                                              .replace("{my_previous_response}", res_b_init)\
                                              .replace("{peer_responses}", peers_for_b)\
                                              .replace("{transcript}", transcript)\
                                              .replace("{score}", score_ctx)\
                                              .replace("{feedback}", feedback_ctx)

            res_b_final, tok_b_final = self.call_novita(p_reflect_b)
            self.df.at[index, 'rec_b_final'] = res_b_final
            self.df.at[index, 'tokens_b_final'] = tok_b_final
            
            peers_for_c = f"--- Peer A (Grammar) ---\n{res_a_init}\n\n--- Peer B (Vocab) ---\n{res_b_init}"
            p_reflect_c = self.critique_prompt.replace("{agent_role}", "Agent C (Conversation Expert)")\
                                              .replace("{my_previous_response}", res_c_init)\
                                              .replace("{peer_responses}", peers_for_c)\
                                              .replace("{transcript}", transcript)\
                                              .replace("{score}", score_ctx)\
                                              .replace("{feedback}", feedback_ctx)

            res_c_final, tok_c_final = self.call_novita(p_reflect_c)
            self.df.at[index, 'rec_c_final'] = res_c_final
            self.df.at[index, 'tokens_c_final'] = tok_c_final
            
            # --- PHASE 3 ---
            p_judge = self.judge_prompt.replace("{agent_a_final}", res_a_final)\
                                       .replace("{agent_b_final}", res_b_final)\
                                       .replace("{agent_c_final}", res_c_final)\
                                       .replace("{transcript}", transcript)
            qwen_json, raw_text, qwen_tokens = self.call_judge(p_judge)
            
            if qwen_json:
                self.df.at[index, 'final_recommendation'] = json.dumps(qwen_json)
                self.df.at[index, 'judge_response'] = raw_text
                self.df.at[index, 'tokens_judge'] = qwen_tokens
            
            self.df.to_csv(self.output_csv_path, index=False)

if __name__ == "__main__":
    processor = RecommendHomoMADProcessor()
    processor.process()
