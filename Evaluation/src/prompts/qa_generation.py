question_answer_prompt = """
You are tasked with generating a fact-based question and its corresponding detailed answer using only the information provided in the context.

Follow these steps:
1. Read the context carefully and extract the key details, facts, and arguments.
2. Formulate a question that is not overly direct but encourages deeper analysis of the context. Avoid questions that result in very short answers.
3. Generate a detailed answer that explains the context and provides comprehensive insights, ensuring it is directly based on the provided context.
4. Both the question and answer must be fact-based and grounded solely in the information from the context.

Below are some examples (delimited by triple backticks):
#Example 1: Context: The industrial revolution brought significant changes in technology, economy, and society. Innovations like the steam engine and mechanized textile production transformed manufacturing, leading to rapid urbanization and the emergence of new labor practices. Furthermore, advancements in transportation and communication connected distant regions, accelerating economic and social transformations. 
## Generated Question: How did the technological innovations of the industrial revolution contribute to urbanization and transform traditional labor practices? 
## Generated Answer: The technological innovations, such as the steam engine and mechanized production methods, centralized manufacturing in urban areas, leading to mass migration from rural regions. This shift not only transformed labor practices by replacing traditional agrarian work with factory-based jobs but also integrated various regions through improved transportation and communication, thus reshaping the societal landscape.

#Example 2: Context: In recent years, renewable energy has emerged as a critical solution to mitigate climate change and reduce dependency on fossil fuels. Significant advancements in solar, wind, and hydroelectric power have made renewable energy sources more viable. However, challenges such as energy storage, grid integration, and intermittent supply persist, requiring continuous innovation and supportive policies. 
## Generated Question: What factors are driving the adoption of renewable energy, and what challenges must be overcome to ensure its widespread implementation? 
## Generated Answer: The adoption of renewable energy is primarily driven by the need to reduce greenhouse gas emissions and decrease reliance on fossil fuels, bolstered by advancements in technologies like solar panels, wind turbines, and hydroelectric systems. Nonetheless, issues such as efficient energy storage, seamless grid integration, and managing intermittent energy supply remain significant challenges that need to be addressed through ongoing technological innovation and robust policy support.

Now, here is the context:

Context: {context}

Please return your answer in the following JSON format:
{{
  "question": "insert generated question here",
  "answer": "insert generated answer here"
}}
Return Output
"""
