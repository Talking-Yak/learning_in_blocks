import os
import pandas as pd
import json
import time
import re
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

# Load environment variables
cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")
env_path = os.path.join(cwd, '.env')
if os.path.exists(env_path):
    print(f".env file found at {env_path}")
else:
    print(".env file NOT found. Searching...")

load_dotenv(dotenv_path=env_path)
print(f"NOVITA_API_KEY present: {'NOVITA_API_KEY' in os.environ}")

class RecommendMADProcessor:
    def __init__(self):
        # Configure Novita AI API
        novita_key = os.getenv('NOVITA_API_KEY')
        if not novita_key:
            raise ValueError("NOVITA_API_KEY not found in environment variables.")
        
        self.novita_client = OpenAI(
            api_key=novita_key,
            base_url="https://api.novita.ai/openai"
        )
        
        # Models for each agent
        self.model_a = "google/gemma-3-27b-it" 
        self.model_b = "qwen/qwen3-30b-a3b-fp8"
        self.model_c = "openai/gpt-oss-20b"
        
        # Configure Gemini API (Judge)
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found.")
            
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gemini_model = 'gemini-2.5-flash'

        # File paths
        self.source_csv_path = "asset/dd_processed.csv"
        self.output_csv_path = "asset/MAD/recommendMAD_output.csv" 
        
        self.prompt_dir = "asset/MAD/recommendMAD/prompt/"
        
        # Context Data Paths
        self.grammar_csv_path = "asset/grammar_flow.csv"
        self.vocab_json_path = "asset/vocab.json"

        # Define output columns
        self.final_columns = [
            'learnerId', 
            'conversationHistoryCleaned', 
            'final_recommendation', # The consensus output
            'rec_a_initial', 'rec_b_initial', 'rec_c_initial',
            'rec_a_final', 'rec_b_final', 'rec_c_final',
            'gemini_judge_response',
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
        
        print("Loading Grammar Context...")
        df_grammar = pd.read_csv(self.grammar_csv_path)
        # Expected cols: skill_no, grammar_skills, skill_description...
        grammar_list = []
        for _, row in df_grammar.iterrows():
            # Format: "ID <no>: <Skill Name> - <Description>"
            entry = f"ID {row['skill_no']}: {row['grammar_skills']} - {str(row['skill_description'])[:100]}..." # Truncate desc slightly to save tokens if needed? No, kept full for now or minor truncate.
            grammar_list.append(entry)
        self.grammar_context = "\n".join(grammar_list)
        
        # 2. Load Vocab Context
        if not os.path.exists(self.vocab_json_path):
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_json_path}")
        
        print("Loading Vocab Context...")
        with open(self.vocab_json_path, 'r') as f:
            vocab_data = json.load(f)
        
        vocab_list = []
        for topic, words in vocab_data.items():
            # Format: "Topic: <Name>" (We can include words if needed, but context might get large. 
            # The prompt says "knows all the topics". Let's provide topics and maybe sample words or all words if it fits.)
            # User said "All three expert will have access to all the given items." -> imply all content.
            word_str = ", ".join(words)
            entry = f"Topic: {topic}\nWords: {word_str}\n"
            vocab_list.append(entry)
        self.vocab_context = "\n".join(vocab_list)

    def load_prompts(self):
        with open(f"{self.prompt_dir}AgentA_Grammar.txt", 'r') as f:
            self.agent_a_prompt = f.read()
        with open(f"{self.prompt_dir}AgentB_Vocab.txt", 'r') as f:
            self.agent_b_prompt = f.read()
        with open(f"{self.prompt_dir}AgentC_Conversation.txt", 'r') as f:
            self.agent_c_prompt = f.read()
            
        with open(f"{self.prompt_dir}Critique_Instruction.txt", 'r') as f:
            self.critique_prompt = f.read()
        
        with open(f"{self.prompt_dir}Gemini_Judge.txt", 'r') as f:
            self.judge_prompt = f.read()

    def initialize_data(self):
        if os.path.exists(self.output_csv_path):
            print(f"Resuming from {self.output_csv_path}...")
            df = pd.read_csv(self.output_csv_path)
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
                elif col in ['final_recommendation', 'rec_a_initial', 'rec_b_initial', 'rec_c_initial', 'rec_a_final', 'rec_b_final', 'rec_c_final', 'gemini_judge_response']:
                     df[col] = df[col].astype('object')
            df = df[self.final_columns]
            return df
        else:
            print(f"Starting new process from {self.source_csv_path}...")
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

    def call_novita(self, model_name, prompt, temperature=0.7):
        while True:
            try:
                response = self.novita_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content
                
                # Remove <think>...</think> blocks if present
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                token_usage = 0
                if hasattr(response, 'usage') and response.usage:
                    token_usage = response.usage.completion_tokens
                    
                return content, token_usage
            except Exception as e:
                error_str = str(e)
                print(f"  - Novita API Error ({model_name}): {error_str[:100]}...")
                if "429" in error_str:
                    print(f"    Rate limit. Waiting 10s...")
                    time.sleep(10)
                else:
                    time.sleep(5)

    def call_gemini(self, prompt, task_name="Gemini Judge"):
        print(f"  - Requesting {task_name}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json'
                    }
                )
                raw_text = response.text
                
                total_tokens = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    total_tokens = response.usage_metadata.total_token_count
                
                return json.loads(raw_text), raw_text, total_tokens
            except Exception as e:
                print(f"  - Gemini Error (Attempt {attempt+1}): {e}")
                time.sleep(2)
        return None, None, 0

    def process(self):
        total_rows = len(self.df)
        print(f"Processing {total_rows} rows...")
        
        for index, row in self.df.iterrows():
            if pd.notna(row['final_recommendation']):
                 continue

            transcript = row['conversationHistoryCleaned']
            if not isinstance(transcript, str) or not transcript.strip():
                print(f"Skipping row {index}: Empty transcript")
                continue

            print(f"\nProcessing Row {index+1}/{total_rows}")
            
            # --- PHASE 1: DIVERGENT THINKING ---
            print("  Phase 1: Initial Analysis...")
            
            # Common replacements
            # Note: We replace placeholders in the prompt string
            
            # Agent A (Grammar)
            p_a = self.agent_a_prompt.replace("{transcript}", transcript)\
                                     .replace("{grammar_context}", self.grammar_context)\
                                     .replace("{vocab_context}", self.vocab_context)
            res_a_init, tok_a_init = self.call_novita(self.model_a, p_a)
            self.df.at[index, 'rec_a_initial'] = res_a_init
            self.df.at[index, 'tokens_a_init'] = tok_a_init
            print(f"   - Agent A (Grammar) done.")
            
            # Agent B (Vocab)
            p_b = self.agent_b_prompt.replace("{transcript}", transcript)\
                                     .replace("{grammar_context}", self.grammar_context)\
                                     .replace("{vocab_context}", self.vocab_context)
            res_b_init, tok_b_init = self.call_novita(self.model_b, p_b)
            self.df.at[index, 'rec_b_initial'] = res_b_init
            self.df.at[index, 'tokens_b_init'] = tok_b_init
            print(f"   - Agent B (Vocab) done.")
            
            # Agent C (Conversation)
            p_c = self.agent_c_prompt.replace("{transcript}", transcript)\
                                     .replace("{grammar_context}", self.grammar_context)\
                                     .replace("{vocab_context}", self.vocab_context)
            res_c_init, tok_c_init = self.call_novita(self.model_c, p_c)
            self.df.at[index, 'rec_c_initial'] = res_c_init
            self.df.at[index, 'tokens_c_init'] = tok_c_init
            print(f"   - Agent C (Conversation) done.")
            
            self.df.to_csv(self.output_csv_path, index=False)
            
            # --- PHASE 2: REFLECTION ---
            print("  Phase 2: Collaborative Reflection...")
            
            # Agent A Reform
            peers_for_a = f"--- Peer B (Vocab) ---\n{res_b_init}\n\n--- Peer C (Conversation) ---\n{res_c_init}"
            p_reflect_a = self.critique_prompt.replace("{agent_role}", "Agent A (Grammar Expert)")\
                                              .replace("{my_previous_response}", res_a_init)\
                                              .replace("{peer_responses}", peers_for_a)\
                                              .replace("{transcript}", transcript)
            res_a_final, tok_a_final = self.call_novita(self.model_a, p_reflect_a)
            self.df.at[index, 'rec_a_final'] = res_a_final
            self.df.at[index, 'tokens_a_final'] = tok_a_final
            
            # Agent B Reform
            peers_for_b = f"--- Peer A (Grammar) ---\n{res_a_init}\n\n--- Peer C (Conversation) ---\n{res_c_init}"
            p_reflect_b = self.critique_prompt.replace("{agent_role}", "Agent B (Vocabulary Expert)")\
                                              .replace("{my_previous_response}", res_b_init)\
                                              .replace("{peer_responses}", peers_for_b)\
                                              .replace("{transcript}", transcript)
            res_b_final, tok_b_final = self.call_novita(self.model_b, p_reflect_b)
            self.df.at[index, 'rec_b_final'] = res_b_final
            self.df.at[index, 'tokens_b_final'] = tok_b_final
            
            # Agent C Reform
            peers_for_c = f"--- Peer A (Grammar) ---\n{res_a_init}\n\n--- Peer B (Vocab) ---\n{res_b_init}"
            p_reflect_c = self.critique_prompt.replace("{agent_role}", "Agent C (Conversation Expert)")\
                                              .replace("{my_previous_response}", res_c_init)\
                                              .replace("{peer_responses}", peers_for_c)\
                                              .replace("{transcript}", transcript)
            res_c_final, tok_c_final = self.call_novita(self.model_c, p_reflect_c)
            self.df.at[index, 'rec_c_final'] = res_c_final
            self.df.at[index, 'tokens_c_final'] = tok_c_final
            
            print("   - All agents refined.")
            self.df.to_csv(self.output_csv_path, index=False)
            
            # --- PHASE 3: CONSENSUS ---
            print("  Phase 3: The Consensus...")
            
            p_judge = self.judge_prompt.replace("{agent_a_final}", res_a_final)\
                                       .replace("{agent_b_final}", res_b_final)\
                                       .replace("{agent_c_final}", res_c_final)\
                                       .replace("{transcript}", transcript)
            
            gemini_json, raw_text, gemini_tokens = self.call_gemini(p_judge, "Gemini Consensus")
            
            if gemini_json:
                print(f"   - Recommendation Found: {gemini_json}")
                self.df.at[index, 'final_recommendation'] = json.dumps(gemini_json)
                self.df.at[index, 'gemini_judge_response'] = raw_text
                self.df.at[index, 'tokens_judge'] = gemini_tokens
            else:
                print("   - Failed to get consensus from Gemini.")
            
            self.df.to_csv(self.output_csv_path, index=False)
            print("  - Saved.")

        print("\nProcessing Complete.")

if __name__ == "__main__":
    processor = RecommendMADProcessor()
    processor.process()
