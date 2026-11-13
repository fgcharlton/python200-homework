# Task 1: Setup and System Prompt
from dotenv import load_dotenv
from openai import OpenAI
import json

if load_dotenv():
    print("Successfully loaded api key")

client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# Write a system prompt that sets up the model as a job application coach
message = [
    {"role": "system",
    "content": "You are a patient, encouraging Job Coach who helps job seekers find a job as a data analyst in a health setting (e.g., healthcare, public health). You will only help with job application materials, remind the user to review and edit any outputs before submitting an application, and acknowledge that output may not fit the user's industry norms (so the user should use their own judgement)."}
]

# I made sure the personality was clear to ensure the system knew that they would help the job seeker with their application materials. I also chose to make the Job Coach encouraging as searching for jobs can be difficult at times.

print("System's Personality: ", message)

# The prompt seems clear and specific enough to help with a job search process.

# Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Do not wrap the JSON in Markdown code fences or add any text before or after it. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages, temperature = 1.0)

    print("Raw response: ", response)

    # Parse JSON
    try:
        result = json.loads(response)
        for item in result:
            print("Original Bullet Point:\n", item["original"])
            print("Improved Bullet Point:\n", item["improved"])
        return result 
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        return response 

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# Did json.loads() succeed without raising an error? If not, try adding "Respond ONLY with valid JSON, no other text." to your prompt. The model sometimes adds a preamble like "Here is the JSON:" that breaks the parser.
# It did not load successfully at first because it added the preamble prior.

# Are both the original and improved versions printing clearly for each bullet?
# Yes, i see the original bullet point then the improved bullet point.

# Do the improvements feel meaningfully better, or are they just rearranged words? If the output is weak, try making your prompt more specific about what "strong" looks like.
# I believe that the improvements are meaninful better. The improved bullet points expand on the original bullet points but show the impact of the work done. This will allow employeers to not only see what work has been done, but how that work has improved the overall workplace.

# Task 3: Cover Letter Generator 
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write a strong cover letter with opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages, temperature = 1.0)

    return response

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Pandas."

generate_cover_letter(job_title, background)

# Why did you choose those particular examples? 
# The examples were pre-generated by the assignment. However, they show clear, confident, and specific examples of cover letters for people that are in a career transition.

# What does the few-shot pattern help control in the output?
# A few-shot gives multiple examples so the model can recognize a pattern and handles the task more reliably. It helps with nuanced, specific tasks like writing a cover letter.

# Does the output feel tailored to the specific person, or is it generic? (Phrases like "I am excited to bring my unique skills..." are a red flag.)
# The output does feel tailored to the specific person, pulling in their unique skilllset as a middle school math teacher as well as their new found skills in creating data pipelines. It went just in-depth enough to show how both those perspectives can contribute to the role and company.

# Does it avoid inventing credentials the user didn't mention?
# It does seem to avoid an credentials that the user didn't mention. 

# Try changing the job title and background to something very different and see if the output adapts. If it sounds the same regardless of input, your prompt may not be specific enough.
job_title2 = "Epidemiologist"
background2 = "Four years of experience as an Epidemiologist; recently completed \
a Python course and expanded data analysis skills using Python."

generate_cover_letter(job_title2, background2)

# The new title and background did shift. The model was able to accurately convey what an Epidemiologist is, as well as how Python could help expand that skillset. I could see it being helpful to expand the prompt to include specific job titles the job seeker is looking for, so that the prompt can more accurately make the connection between current experience, new skills, and next job.

# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("I can't help with that - please rephrase.")
        return False
    return True

# Input that shouldn't pass 
print("Unsafe statement: ", is_safe("I want to kill my neighbor."))

# Input that should pass 
print("Safe statement: ", is_safe("I want to water my neighbor's plant."))

# Does your flagged test case actually get caught? If not, try a more explicit phrase.
# Yes, the unsafe statement 'I want to kill my neighbor' was flagged as unsafe and the user would need to input another safe statement. 

# Does your safe test case pass without triggering any warning?
# Yes, the safe statement 'I want to water my neighbor's plant' passes without triggering any warning. It simply outputs 'True'.

# What happens if you test a borderline phrase? Look at result.results[0].categories to see which category was triggered.
result = client.moderations.create(model="omni-moderation-latest", input="I want to kill my presentation")
print("Boderline flagged: ", result.results[0].flagged)
print(result.results[0].categories.model_dump())

# This borderline statement passes as safe. However, it is flagged as 'True' for the 'violence' category. Since the moderations 
# endpoint looks at the entire message context, it passes as 'True' for being safe.

# Task 5: The Chatbot Loop
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": "You are a patient, encouraging Job Coach who helps job seekers find a job as a data analyst in a health setting (e.g., healthcare, public health). You will only help with job application materials, remind the user to review and edit any outputs before submitting an application, and acknowledge that output may not fit the user's industry norms (so the user should use their own judgement)."}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            result = rewrite_bullets(raw_bullets)
            print(result)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_result = generate_cover_letter(job_title, background)
            print(cover_result)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})

            # - Call get_completion(messages)
            reply = get_completion(messages, model="gpt-4o-mini", temperature=0.7)

            # - Print the reply
            print(f"Job Application Helper: {reply}")
            print("(Type 'quit' anytime to end the chat.)\n")

            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": reply})

            pass

if __name__ == "__main__":
    run_chatbot()

# Have a short conversation with your bot (3-4 exchanges) without using the bullet or cover letter features. After each turn, add a temporary print(len(messages)) to confirm the history is growing. Remove it when you're done.
# print(len(messages)) works.

# Ask the bot something from turn 1, then reference it in turn 3 (e.g., give your name in the first message, then ask "what did I tell you my name was?"). If it can't remember, check that you're appending both user and assistant messages to messages after every turn.
# Bot was able to reference something from turn 1. 

# Try triggering the bullet rewriter and cover letter generator from inside the loop and confirm they still work.
# Role: Epidemiologist

# Original Bullet Points Given:
#●	Developed and standardized datasets to streamline dashboard building, analysis, and data sharing, improving data consistency and operational efficiency
#●	Built and optimized an interactive data dashboard to enhance user engagement, data literacy, and accessibility, enabling partners to make informed, data driven decisions 
#●	Compiled and delivered presentations to both technical and non-technical audiences, effectively translating complex data into actionable insights and facilitating informed decision making 
#●	Evaluated surveillance systems and implemented quality improvement measures, enhancing reliability 
#●	Established and led a statewide group working to reduce excessive alcohol use, fostering interdisciplinary partnerships and driving collaboration to support data driven decision making

# New Bullet Points:
# ● Engineered standardized datasets that accelerated dashboard construction and analysis processes, resulting in a 30% increase in data consistency and operational efficiency.
# ● Created and refined an interactive data dashboard, boosting user engagement by 40% and facilitating data literacy, empowering partners to make informed, data-driven decisions.
# ● Delivered dynamic presentations to diverse audiences, converting complex data into actionable insights that improved decision-making processes for 100+ stakeholders.
# ● Assessed and upgraded surveillance systems, integrating quality improvement initiatives that enhanced reliability by 25%.
# ● Founded and directed a statewide coalition dedicated to reducing excessive alcohol use, cultivating interdisciplinary partnerships that accelerated data-driven decision-making initiatives.

# Some bullet points were improved, however, I would need to adapt and ensure they are staying true to the original work. 

# Type quit and confirm the bot exits cleanly.
# Exits cleanly with an encouraging message of 'Good luck with your applications!' 

# Task 6: Ethics Reflection
# Option A: Comment block at the bottom of project_05.py
"""
What could go wrong if a job-seeker submitted the bot's output directly — 
without reviewing it — to a real employer?
AI-generagted content should never be treated as final. The user must review the content
to check for hallucincations, inaccuracies, and bias before using it. If a job-seeker submitted the bot's 
output directly, without reviewing, it could cost them the chance to interview or secure them an interview, 
but cost them the job if the AI overstated the job-seeker's experience and skills. 

What is one guardrail you would add if you were deploying this tool 
professionally? (A guardrail is any design choice that reduces the 
chance of harm — a UI warning, a moderation filter, a usage policy, 
a disclaimer, or something else entirely.)
I would add a disclaimer that shares that a user should review the bot's output before using it in either their
resume or cover letter. This would allow the user to know that user review is needed to ensure accuracy. In addition,
review can help ensure that the output is in alignment with both the user's experience as well as the job application's 
expectations. 
"""
