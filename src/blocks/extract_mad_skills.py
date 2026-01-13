import pandas as pd
import json
import re
import os

def load_grammar_mapping(grammar_file_path):
    """
    Loads the grammar skills mapping from the CSV file.
    Returns a dictionary mapping skill_no (int) to grammar_skills (str).
    """
    try:
        df = pd.read_csv(grammar_file_path)
        # Ensure skill_no is int and create dictionary
        mapping = pd.Series(df.grammar_skills.values, index=df.skill_no).to_dict()
        return mapping
    except Exception as e:
        print(f"Error loading grammar mapping: {e}")
        return {}

def clean_json_string(s):
    """
    Cleans the JSON string from Markdown code blocks or other artifacts.
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Remove markdown code blocks if present
    match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
    if match:
        s = match.group(1)
    elif s.startswith("```") and s.endswith("```"):
         s = s[3:-3].strip()
    
    return s

def extract_skills_markdown(row, grammar_mapping):
    """
    Extracts grammar_skills and vocab_skills from the gemini_judge_response column.
    Maps grammar skill IDs to their names.
    Returns a formatted markdown string.
    """
    response_str = row.get('gemini_judge_response')
    
    # helper for empty/error state
    empty_md = "grammar_skills:\n\n___\n\nvocab_skills:"

    if pd.isna(response_str):
        return empty_md

    cleaned_response = clean_json_string(response_str)
    
    try:
        data = json.loads(cleaned_response)
        
        # Extract grammar IDs and map to names
        grammar_ids = data.get('grammar_skills', [])
        # We need to handle potential parsing issues for IDs slightly robustly
        grammar_names = []
        for gid in grammar_ids:
            # Clean gid just in case, though json.loads usually handles types
            if isinstance(gid, (int, str)) and str(gid).isdigit():
                gname = grammar_mapping.get(int(gid), f"Unknown ID {gid}")
                grammar_names.append(gname)
        
        # Extract vocab skills
        vocab_skills = data.get('vocab_skills', [])
        
        # Build Markdown String
        # grammar_skills:
        # 1. ...
        # 2. ...
        lines = ["grammar_skills:"]
        for i, name in enumerate(grammar_names, 1):
             lines.append(f"{i}. {name}")
        
        lines.append("")
        lines.append("___")
        lines.append("")
        
        lines.append("vocab_skills:")
        for i, name in enumerate(vocab_skills, 1):
            lines.append(f"{i}. {name}")
            
        return "\n".join(lines)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for learner {row.get('learnerId')}: {e}")
        return empty_md
    except Exception as e:
        print(f"Unexpected error for learner {row.get('learnerId')}: {e}")
        return empty_md

def main():
    # File paths
    base_dir = '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks'
    input_csv_path = os.path.join(base_dir, 'asset/MAD/recommendMAD_output.csv')
    grammar_csv_path = os.path.join(base_dir, 'asset/grammar_flow.csv')
    output_csv_path = os.path.join(base_dir, 'asset/MAD/recommendMAD_skills_extracted.csv')

    # Load grammar mapping
    print("Loading grammar mapping...")
    grammar_mapping = load_grammar_mapping(grammar_csv_path)
    print(f"Loaded {len(grammar_mapping)} grammar skills.")

    # Load input CSV
    print(f"Loading input CSV from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"File not found: {input_csv_path}")
        return

    expected_col = 'gemini_judge_response'
    if expected_col not in df.columns:
        print(f"Column '{expected_col}' not found in input CSV.")
        return

    # Apply extraction
    print("Extracting skills...")
    # We will create a new DataFrame for the output
    output_df = pd.DataFrame()
    
    # If using learnerId as identifier (recommended even if not explicitly asked, to keep track)
    if 'learnerId' in df.columns:
        output_df['learnerId'] = df['learnerId']
    
    # Extract the info and store in a new column 'extracted_skills'
    # The user asked to "make it a JSON", so we store the JSON string.
    output_df['extracted_info'] = df.apply(lambda row: extract_skills_markdown(row, grammar_mapping), axis=1)

    # Save to CSV
    print(f"Saving output to {output_csv_path}...")
    output_df.to_csv(output_csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
