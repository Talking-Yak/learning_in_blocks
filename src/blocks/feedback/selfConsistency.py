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

class SelfConsistencyProcessor:
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
        
        # Configure Gemini API (Consensus)
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found.")
            
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gemini_model = 'gemini-2.5-flash'

        # File paths
        self.source_csv_path = "asset/dd_processed.csv"
        self.output_csv_path = "asset/selfConsistency/selfConsistencyNT.csv" 
        self.score_prompt_path = "asset/selfConsistency/prompt/selfConsistencyScore.txt"
        self.consensus_prompt_path = "asset/selfConsistency/prompt/consensusPrompt.txt"
        self.feedback_prompt_path = "asset/selfConsistency/prompt/selfConsistencyFeedback.txt"
        
        # Load logic
        self.load_prompts()
        self.df = self.initialize_data()

    def load_prompts(self):
        with open(self.score_prompt_path, 'r') as f:
            self.score_prompt_template = f.read()
        
        with open(self.consensus_prompt_path, 'r') as f:
            self.consensus_template = f.read()
            
        with open(self.feedback_prompt_path, 'r') as f:
            self.feedback_prompt_template = f.read()

    def initialize_data(self):
        """
        Load existing progress from output file if it exists, otherwise load from source.
        Does NOT modify source file.
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
        
        # 1. Remove <think> blocks entirely
        if "<think>" in json_str:
            # Remove content between tags
            json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
            # Remove any stray tags
            json_str = json_str.replace("<think>", "").replace("</think>", "")
            
        # 2. Clean Markdown
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
            
        return json_str.strip()

    def get_qwen_responses(self, transcript):
        """Get 3 raw responses from Qwen + Token Usage"""
        prompt = self.score_prompt_template.replace("{conversationHistoryCleaned}", transcript)
        responses = []
        token_usages = []
        
        for i in range(3):
            print(f"  - Requesting Qwen response {i+1}/3...")
            content, tokens = self.call_qwen_raw(prompt, temperature=0.7)
            responses.append(content)
            token_usages.append(tokens)
            # Short sleep to avoid burst rate limits
            time.sleep(1) 
            
        return responses, token_usages

    def get_gemini_consensus(self, transcript, qwen_responses):
        """
        Uses Gemini 2.5 Flash to analyze the 3 Qwen responses and the transcript
        to produce a final consensus score in valid JSON.
        """
        # Format the 3 responses for the prompt
        responses_text = ""
        for i, resp in enumerate(qwen_responses):
            responses_text += f"\n--- Response {i+1} ---\n{resp}\n"
            
        prompt = self.consensus_template.replace("{transcript}", transcript)
        prompt = prompt.replace("{responses}", responses_text)
        
        print("  - Requesting Gemini Consensus...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use generate_content for Gemini
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json'
                    }
                )
                
                # Parse JSON
                json_str = response.text
                return json.loads(json_str) 
                
            except Exception as e:
                print(f"  - Gemini Error (Attempt {attempt+1}): {e}")
                time.sleep(2)
        
        return None

    def get_qwen_feedback(self, transcript, consensus_score):
        """Get feedback from Qwen using the consensus score"""
        prompt = self.feedback_prompt_template.replace("{conversationHistoryCleaned}", transcript)
        # Ensure consensus_score is a valid JSON string of the scores
        prompt = prompt.replace("{score}", json.dumps(consensus_score))
        
        print("  - Requesting Qwen Feedback...")
        
        while True:
            raw_response, token_usage = self.call_qwen_json(prompt, temperature=0.5)
            try:
                # Clean clean tags, then parse
                clean_response = self.clean_json_string(raw_response)
                return json.loads(clean_response), token_usage
            except json.JSONDecodeError:
                print("    Invalid JSON feedback. Retrying...")
                # Could print repr(raw_response) here to debug if needed
                time.sleep(2)

    def process(self):
        total_rows = len(self.df)
        print(f"Processing {total_rows} rows...")
        
        for index, row in self.df.iterrows():
            # Resume logic: Skip if we have score AND feedback
            current_score = row['score']
            current_feedback = row['feedback']
            
            if pd.notna(current_score) and pd.notna(current_feedback) and str(current_score).strip() != "":
                 continue

            transcript = row['conversationHistoryCleaned']
            if not isinstance(transcript, str) or not transcript.strip():
                print(f"Skipping row {index}: Empty transcript")
                continue

            print(f"\nProcessing Row {index+1}/{total_rows}")
            
            # 1. Get 3 Raw Responses from Qwen (No JSON enforcement)
            responses, token_usages = self.get_qwen_responses(transcript)
            
            # Save individual responses and tokens to CSV
            if len(responses) >= 1: 
                self.df.at[index, 'response1'] = responses[0]
                self.df.at[index, 'tokens_response1'] = token_usages[0]
                
            if len(responses) >= 2: 
                self.df.at[index, 'response2'] = responses[1]
                self.df.at[index, 'tokens_response2'] = token_usages[1]
                
            if len(responses) >= 3: 
                self.df.at[index, 'response3'] = responses[2]
                self.df.at[index, 'tokens_response3'] = token_usages[2]
                
            self.df.to_csv(self.output_csv_path, index=False) # Save checkpoint

            # 2. Get Consensus Score from Gemini (Score ONLY)
            consensus_result = self.get_gemini_consensus(transcript, responses)
            
            if consensus_result:
                score_data = consensus_result.get('score', consensus_result) # Handle if it's nested or direct
                print(f"  - Gemini Consensus Score: {score_data}")
                self.df.at[index, 'score'] = json.dumps(score_data)
                
                # 3. Get Feedback from Qwen (using Gemini's Score)
                feedback_data, feedback_tokens = self.get_qwen_feedback(transcript, score_data)
                self.df.at[index, 'feedback'] = json.dumps(feedback_data)
                self.df.at[index, 'tokens_feedback'] = feedback_tokens
                
            else:
                print("  - Failed to get consensus from Gemini.")
            
            # Save final row state
            self.df.to_csv(self.output_csv_path, index=False)
            print("  - Saved.")

        print("\nProcessing Complete.")

if __name__ == "__main__":
    processor = SelfConsistencyProcessor()
    processor.process()