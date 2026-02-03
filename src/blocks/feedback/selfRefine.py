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

class SelfRefineProcessor:
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
        
        # Configure Gemini API (Formatting)
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found.")
            
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gemini_model = 'gemini-2.5-flash'

        # File paths
        self.source_csv_path = "asset/CSVs/dd_processed.csv"
        self.output_csv_path = "asset/CSVs/selfRefineNT.csv" 
        
        self.score_prompt_path = "asset/selfRefine/selfRefineScore.txt"
        self.review_prompt_path = "asset/selfRefine/selfRefineReview.txt"
        self.format_prompt_path = "asset/selfRefine/formattingPrompt.txt"
        self.feedback_prompt_path = "asset/selfRefine/selfRefineFeedback.txt"
        
        # Load logic
        self.load_prompts()
        self.df = self.initialize_data()

    def load_prompts(self):
        with open(self.score_prompt_path, 'r') as f:
            self.score_prompt_template = f.read()
            
        with open(self.review_prompt_path, 'r') as f:
            self.review_prompt_template = f.read()
        
        with open(self.format_prompt_path, 'r') as f:
            self.format_template = f.read()
            
        with open(self.feedback_prompt_path, 'r') as f:
            self.feedback_prompt_template = f.read()

    def initialize_data(self):
        """
        Load existing progress from output file if it exists, otherwise load from source.
        """
        required_cols = ['score', 'feedback', 'response1', 'response2', 'response3', 
                         'tokens_response1', 'tokens_response2', 'tokens_response3', 'tokens_feedback']
        
        if os.path.exists(self.output_csv_path):
            print(f"Resuming from {self.output_csv_path}...")
            df = pd.read_csv(self.output_csv_path)
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
            return df
        else:
            print(f"Starting new process from {self.source_csv_path}...")
            if not os.path.exists(self.source_csv_path):
                raise FileNotFoundError(f"Input CSV not found at {self.source_csv_path}")
            df = pd.read_csv(self.source_csv_path)
            
            # Initialize new columns
            for col in required_cols:
                df[col] = None
                df[col] = df[col].astype('object')
                
            # Save immediately to establish the output file
            df.to_csv(self.output_csv_path, index=False)
            return df

    def call_qwen_raw(self, prompt, temperature=0.7):
        """
        Calls Qwen API to get a raw response (text).
        Retries on errors.
        Returns: content, token_usage
        """
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
    
    def call_qwen_json(self, prompt, temperature=0.5):
        """
        Calls Qwen API to get a JSON response (for feedback).
        """
        while True:
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.qwen_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=temperature,
                    response_format={ "type": "json_object" } 
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

    def clean_json_string(self, json_str):
        """Clean markdown tags and remove <think> blocks from JSON string"""
        if "<think>" in json_str:
            json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
            json_str = json_str.replace("<think>", "").replace("</think>", "")
            
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
            
        return json_str.strip()

    def get_sequential_responses(self, transcript):
        """
        Step 1: Initial Score
        Step 2: Refine Score based on Step 1
        Step 3: Refine Score based on Step 2
        """
        
        # --- Step 1 ---
        print("  - Step 1: Initial Scoring...")
        prompt1 = self.score_prompt_template.replace("{conversationHistoryCleaned}", transcript)
        resp1, tokens1 = self.call_qwen_raw(prompt1, temperature=0.7)
        time.sleep(1)

        # --- Step 2 ---
        print("  - Step 2: Refinement 1...")
        # Inject previous response into review prompt
        prompt2 = self.review_prompt_template.replace("{conversationHistoryCleaned}", transcript)
        prompt2 = prompt2.replace("{previous_response}", resp1)
        resp2, tokens2 = self.call_qwen_raw(prompt2, temperature=0.7)
        time.sleep(1)

        # --- Step 3 ---
        print("  - Step 3: Refinement 2...")
        # Inject previous response (resp2) into review prompt
        prompt3 = self.review_prompt_template.replace("{conversationHistoryCleaned}", transcript)
        prompt3 = prompt3.replace("{previous_response}", resp2)
        resp3, tokens3 = self.call_qwen_raw(prompt3, temperature=0.7)
        time.sleep(1)
        
        return (resp1, tokens1), (resp2, tokens2), (resp3, tokens3)

    def get_gemini_formatting(self, input_data):
        """
        Uses Gemini to format the final Qwen response (resp3) into valid JSON.
        """
        prompt = self.format_template.replace("{input_data}", input_data)
        
        print("  - Requesting Gemini Formatting...")
        
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
                return json.loads(response.text) 
            except Exception as e:
                print(f"  - Gemini Error (Attempt {attempt+1}): {e}")
                time.sleep(2)
        
        return None

    def get_qwen_feedback(self, transcript, final_score):
        """Get feedback from Qwen using the final formatted score"""
        prompt = self.feedback_prompt_template.replace("{conversationHistoryCleaned}", transcript)
        prompt = prompt.replace("{score}", json.dumps(final_score))
        
        print("  - Requesting Qwen Feedback...")
        
        while True:
            raw_response, token_usage = self.call_qwen_json(prompt, temperature=0.5)
            try:
                clean_response = self.clean_json_string(raw_response)
                return json.loads(clean_response), token_usage
            except json.JSONDecodeError:
                print("    Invalid JSON feedback. Retrying...")
                time.sleep(2)

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
            
            # 1. Get Sequential Responses
            (r1, t1), (r2, t2), (r3, t3) = self.get_sequential_responses(transcript)
            
            # Save to DF
            self.df.at[index, 'response1'] = r1
            self.df.at[index, 'tokens_response1'] = t1
            self.df.at[index, 'response2'] = r2
            self.df.at[index, 'tokens_response2'] = t2
            self.df.at[index, 'response3'] = r3
            self.df.at[index, 'tokens_response3'] = t3
            
            self.df.to_csv(self.output_csv_path, index=False)

            # 2. Format Final Response with Gemini
            formatted_result = self.get_gemini_formatting(r3)
            
            if formatted_result:
                score_data = formatted_result.get('score', formatted_result)
                print(f"  - Final Score: {score_data}")
                self.df.at[index, 'score'] = json.dumps(score_data)
                
                # 3. Get Feedback
                feedback_data, feedback_tokens = self.get_qwen_feedback(transcript, score_data)
                self.df.at[index, 'feedback'] = json.dumps(feedback_data)
                self.df.at[index, 'tokens_feedback'] = feedback_tokens
            else:
                print("  - Failed to format response with Gemini.")
            
            self.df.to_csv(self.output_csv_path, index=False)
            print("  - Saved.")

        print("\nProcessing Complete.")

if __name__ == "__main__":
    processor = SelfRefineProcessor()
    processor.process()