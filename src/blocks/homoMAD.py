import os
import pandas as pd
import json
import time
import re
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

# Load environment variables
load_dotenv()

class HomoMADProcessor:
    def __init__(self):
        # Configure Novita AI API (Qwen)
        novita_key = os.getenv('NOVITA_API_KEY')
        if not novita_key:
            raise ValueError("NOVITA_API_KEY not found in environment variables.")
        
        self.novita_client = OpenAI(
            api_key=novita_key,
            base_url="https://api.novita.ai/openai"
        )
        self.qwen_model_name = "qwen/qwen3-30b-a3b-fp8" 
        
        # Configure Gemini API (Judge & Feedback)
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found.")
            
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gemini_model = 'gemini-2.5-flash'

        # File paths
        self.source_csv_path = "asset/dd_processed.csv"
        self.output_csv_path = "asset/MAD/homoMAD_output.csv" 
        
        self.prompt_dir = "asset/MAD/prompt/"
        
        # Define exact column order to match output file
        self.final_columns = [
            'learnerId', 'tag1', 'tag2', 'tag3', 'lessonCount', 
            'conversationHistory', 'conversationHistoryCleaned', 
            'score', 
            'agent_a_initial', 'agent_b_initial', 'agent_c_initial',
            'agent_a_final', 'agent_b_final', 'agent_c_final',
            'gemini_judge_response',
            'tokens_a_init', 'tokens_b_init', 'tokens_c_init',
            'tokens_a_final', 'tokens_b_final', 'tokens_c_final',
            'tokens_judge',
            'feedback'
        ]
        
        # Load logic
        self.load_prompts()
        self.df = self.initialize_data()

    def load_prompts(self):
        with open(f"{self.prompt_dir}AgentA_Initial.txt", 'r') as f:
            self.agent_a_prompt = f.read()
        with open(f"{self.prompt_dir}AgentB_Initial.txt", 'r') as f:
            self.agent_b_prompt = f.read()
        with open(f"{self.prompt_dir}AgentC_Initial.txt", 'r') as f:
            self.agent_c_prompt = f.read()
            
        with open(f"{self.prompt_dir}Critique_Instruction.txt", 'r') as f:
            self.critique_prompt = f.read()
        
        with open(f"{self.prompt_dir}Gemini_Judge.txt", 'r') as f:
            self.judge_prompt = f.read()

    def initialize_data(self):
        # Columns that need to be ensured exist
        
        if os.path.exists(self.output_csv_path):
            print(f"Resuming from {self.output_csv_path}...")
            df = pd.read_csv(self.output_csv_path)
            # Ensure all columns exist
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
                elif col in ['score', 'feedback', 'gemini_judge_response']:
                     df[col] = df[col].astype('object')
            
            # Reorder
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
            
            # Reorder
            df = df[self.final_columns]
                
            df.to_csv(self.output_csv_path, index=False)
            return df

    def call_qwen(self, prompt, temperature=0.7):
        """Calls Qwen API. Returns content and token usage."""
        while True:
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.qwen_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content
                token_usage = 0
                if hasattr(response, 'usage') and response.usage:
                    token_usage = response.usage.completion_tokens
                    
                return content, token_usage
            except Exception as e:
                error_str = str(e)
                print(f"  - Qwen API Error: {error_str[:100]}...")
                if "429" in error_str:
                    print(f"    Rate limit. Waiting 10s...")
                    time.sleep(10)
                else:
                    time.sleep(5)

    def call_gemini(self, prompt, task_name="Gemini Judge"):
        """Calls Gemini to produce final JSON. Returns (parsed_json, raw_text, total_tokens)"""
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
                
                # Extract token usage
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
            if pd.notna(row['score']) and pd.notna(row['feedback']) and str(row['score']).strip() != "":
                 continue

            transcript = row['conversationHistoryCleaned']
            if not isinstance(transcript, str) or not transcript.strip():
                print(f"Skipping row {index}: Empty transcript")
                continue

            print(f"\nProcessing Row {index+1}/{total_rows}")
            
            # --- PHASE 1: DIVERGENT THINKING ---
            print("  Phase 1: Divergent Thinking...")
            
            # Agent A
            p_a = self.agent_a_prompt.replace("{transcript}", transcript)
            res_a_init, tok_a_init = self.call_qwen(p_a)
            self.df.at[index, 'agent_a_initial'] = res_a_init
            self.df.at[index, 'tokens_a_init'] = tok_a_init
            print("   - Agent A (Grammar) done.")
            
            # Agent B
            p_b = self.agent_b_prompt.replace("{transcript}", transcript)
            res_b_init, tok_b_init = self.call_qwen(p_b)
            self.df.at[index, 'agent_b_initial'] = res_b_init
            self.df.at[index, 'tokens_b_init'] = tok_b_init
            print("   - Agent B (Lexical) done.")
            
            # Agent C
            p_c = self.agent_c_prompt.replace("{transcript}", transcript)
            res_c_init, tok_c_init = self.call_qwen(p_c)
            self.df.at[index, 'agent_c_initial'] = res_c_init
            self.df.at[index, 'tokens_c_init'] = tok_c_init
            print("   - Agent C (Pragmatic) done.")
            
            self.df.to_csv(self.output_csv_path, index=False)
            
            # --- PHASE 2: COLLABORATIVE REFINEMENT ---
            print("  Phase 2: Collaborative Refinement (Debate)...")
            
            # Agent A Reform
            peers_for_a = f"--- Peer B (Lexical) ---\n{res_b_init}\n\n--- Peer C (Pragmatic) ---\n{res_c_init}"
            p_debate_a = self.critique_prompt.replace("{agent_role}", "Agent A (Strict Grammarian)")\
                                             .replace("{my_previous_response}", res_a_init)\
                                             .replace("{peer_responses}", peers_for_a)\
                                             .replace("{transcript}", transcript)
            res_a_final, tok_a_final = self.call_qwen(p_debate_a)
            self.df.at[index, 'agent_a_final'] = res_a_final
            self.df.at[index, 'tokens_a_final'] = tok_a_final
            
            # Agent B Reform
            peers_for_b = f"--- Peer A (Grammar) ---\n{res_a_init}\n\n--- Peer C (Pragmatic) ---\n{res_c_init}"
            p_debate_b = self.critique_prompt.replace("{agent_role}", "Agent B (Lexical Auditor)")\
                                             .replace("{my_previous_response}", res_b_init)\
                                             .replace("{peer_responses}", peers_for_b)\
                                             .replace("{transcript}", transcript)
            res_b_final, tok_b_final = self.call_qwen(p_debate_b)
            self.df.at[index, 'agent_b_final'] = res_b_final
            self.df.at[index, 'tokens_b_final'] = tok_b_final
            
            # Agent C Reform
            peers_for_c = f"--- Peer A (Grammar) ---\n{res_a_init}\n\n--- Peer B (Lexical) ---\n{res_b_init}"
            p_debate_c = self.critique_prompt.replace("{agent_role}", "Agent C (Pragmatic Communicator)")\
                                             .replace("{my_previous_response}", res_c_init)\
                                             .replace("{peer_responses}", peers_for_c)\
                                             .replace("{transcript}", transcript)
            res_c_final, tok_c_final = self.call_qwen(p_debate_c)
            self.df.at[index, 'agent_c_final'] = res_c_final
            self.df.at[index, 'tokens_c_final'] = tok_c_final
            
            print("   - All agents refined.")
            self.df.to_csv(self.output_csv_path, index=False)
            
            # --- PHASE 3: THE RESOLUTION (JUDGE & FEEDBACK) ---
            print("  Phase 3: The Resolution...")
            
            p_judge = self.judge_prompt.replace("{agent_a_initial}", res_a_init)\
                                       .replace("{agent_a_final}", res_a_final)\
                                       .replace("{agent_b_initial}", res_b_init)\
                                       .replace("{agent_b_final}", res_b_final)\
                                       .replace("{agent_c_initial}", res_c_init)\
                                       .replace("{agent_c_final}", res_c_final)\
                                       .replace("{transcript}", transcript)
            
            gemini_json, raw_text, gemini_tokens = self.call_gemini(p_judge, "Gemini Judge")
            
            if gemini_json:
                # Save Scores
                score_data = gemini_json.get('score', {})
                print(f"   - Final Consensus Score: {score_data}")
                self.df.at[index, 'score'] = json.dumps(score_data)
                
                # Save Feedback
                feedback_data = gemini_json.get('feedback', {})
                print(f"   - Personalized Feedback Curated.")
                self.df.at[index, 'feedback'] = json.dumps(feedback_data)
                
                # Save Meta
                self.df.at[index, 'gemini_judge_response'] = raw_text
                self.df.at[index, 'tokens_judge'] = gemini_tokens
            else:
                print("   - Failed to get consensus from Gemini.")
            
            self.df.to_csv(self.output_csv_path, index=False)
            print("  - Saved.")

        print("\nProcessing Complete.")

if __name__ == "__main__":
    processor = HomoMADProcessor()
    processor.process()