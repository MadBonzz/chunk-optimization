evaluation_prompt = """
You are an evaluator determining whether a given context contains enough information for generating meaningful questions.  
Your evaluation should follow these criteria:  

The evaluation should satisfy the rules below:  
{rules}

Follow these steps to evaluate the context:  
{guidelines}

Here are some examples (delimited by triple backticks):  
{examples}

Now here is the context (delimited by triple quotes):  

Context: \"\"\"{context}\"\"\"  

Your Analysis:
- **Reasoning**: (Yes/No, explain why)  
- **Final Evaluation Score (0-1):**  
"""

rules = """ 
- The context must present a clear subject or main idea.  
- The context must include specific details, facts, or examples.  
- The context must contain claims, arguments, or explanations that could be questioned.  
- The context must have sufficient depth or complexity to allow meaningful questions to be generated."""

guidelines = """ 
1. **Analyze the subject clarity** – Does the context introduce a well-defined topic?  
2. **Check for depth** – Does the context contain enough details, reasoning, or claims?  
3. **Assess questionability** – Can a meaningful question be generated from the context?  
4. **Assign a rating (0-1 scale) and justify it** based on the criteria.  

### Scoring Scale:  
- **0.0 - 0.3**: The context is too vague or generic, making it impossible to form a meaningful question.  
- **0.4 - 0.6**: The context has some useful information but lacks depth or supporting details.  
- **0.7 - 1.0**: The context is rich with details and allows for meaningful question generation."""

examples = """
# Example 1 (Positive, High Score):  
## Context: "The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days."  
## Reasoning: The context presents a well-defined topic (Earth's orbit), provides specific details (elliptical path, 365.25 days), and allows for meaningful questions (e.g., "Why is Earth's orbit elliptical?").  
## Evaluation: **0.9**  

# Example 2 (Negative, Low Score):  
## Context: "Apples are a type of fruit."  
## Reasoning: This statement is too general. It lacks details, claims, or explanations that could be questioned.  
## Evaluation: **0.2**  

# Example 3 (Positive, Medium-High Score):  
## Context: "Water boils at 100°C at sea level, but the boiling point decreases at higher altitudes due to lower atmospheric pressure."  
## Reasoning: The statement provides a clear subject (boiling point), specific details (temperature, altitude effects), and a scientific principle that can be questioned.  
## Evaluation: **0.8**  

# Example 4 (Negative, Mid-Low Score):  
## Context: "The Renaissance was a time of cultural rebirth."  
## Reasoning: While the topic is clear (Renaissance), the statement lacks specific details or supporting explanations. It does not provide enough depth to generate a meaningful question beyond general knowledge.  
## Evaluation: **0.4**  
"""
