import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_answer(question, file_content=None):
    """
    Get answer to an assignment question using an LLM.
    
    Parameters:
    - question: The assignment question
    - file_content: Optional processed file content
    
    Returns:
    - Answer string
    """
    # Prepare the prompt
    prompt = f"Answer the following Data Science assignment question: {question}\n\n"
    
    if file_content is not None:
        # If it's a DataFrame
        if hasattr(file_content, 'to_string'):
            file_content_str = file_content.to_string()
            prompt += f"The file content is:\n{file_content_str}\n\n"
        else:
            prompt += f"The file content is:\n{file_content}\n\n"
    
    prompt += "Provide only the direct answer with no explanations or additional text."
    
    # Call the LLM API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another appropriate model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides direct answers to Data Science assignment questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # Lower temperature for more deterministic responses
    )
    
    # Extract the answer
    answer = response.choices[0].message.content.strip()
    return answer