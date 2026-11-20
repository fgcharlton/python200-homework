# Part 1: Warmup Exercises
# Initial setup
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from scipy import stats
# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# Q1: Define the following Python function (use as written)
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Write the JSON schema dictionary that describes this function to LLM (inclue name, description, amd parameters)
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
            'parameters': {
                'type': 'object',
                'required': ['celsius'],
                'properties': {
                    'celsius': {'type': 'number',
                    'description': 'The temperature to convert in celsius.'},
                },
            },
        },
    }
]
print('Tools listed defined with one tool: celsius_to_fahrenheit')

# Call the function directly (not through an agent) with 0, 100, and -40
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q2: Copy the run_agent function 
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print('Tools list defined with one tool: get_current_time')

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Before calling the tool, add a comment block that predicts:
"""
Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") 
trigger a tool call? Why or why not?
No, because the current call does not have a tool to support the user asking the model to convert a temperature
from celsius to fahrenheit. It only has a tool to support the user asking about the current time. Run_agent has set 
tool_choice to 'auto', so the model is not running any special tool-selection algorithm. It should simply read 
the conversation and the tool descriptions that I provided above (through the JSON schema) and use its normal
language-model reasoning to see if there is a tool that answers the question.

How many API calls will be made to answer this query?
The API should be called once. The first call will read the conversation, decide whether a tool is
needed, and return a message. Since, the model does not have the tool currently, it will answer directly 
(either converting the temperature or not) and not call a tool.
"""

# Call run_agent("Convert 100 degress Celsius to Fahrenheit") and print result
answer_with_agent = run_agent("Convert 100 degress Celsius to Fahrenheit")
print(answer_with_agent)

# Was my prediction correct? 
"""Yes my predictions were mostly correct. It did not call a tool, but was still able to call a formula and convert the temperature.
I wasn't sure that the model was able to answer that question with the given information. """

# Q3: Extend agent to support both tools
tools = [
        {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
            'parameters': {
                'type': 'object',
                'required': ['celsius'],
                'properties': {
                    'celsius': {'type': 'number',
                    'description': 'The temperature to convert in celsius.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print('Tools list defined with two tools: get_current_time and celsius_to_fahrenheit')

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant with two tools.
                    Use celsius_to_fahrenheit when a user asks to convert a temperature from celsius to fahrenheit.
                    Use get_current_time when a user asks for the current time. For anything else, answer directly.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                args = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(args['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Test the extended agent on both queries:
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# What tool was called and why?
"""
The celsius_to_fahrenheit tool was called because the user asked for the agent to convert 37 degrees celsius to fahrenheit. 
"""

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# What tool was called and why?
"""
No tool was needed because the user asked the model a conceptual question about the boiling point of water. 
"""

# --- Lesson 03: Multi-Tool Agent ---
# For Q4-Q6, use the full CsvManager class and run_agent_cycle setup from the lesson (copy them into your 
# file). You will extend them. Make sure bike_commute.csv is in assignments_07/resources/ (see the setup 
# note at the top of this page).

RESOURCES_DIR = Path("resources")
RESOURCES_DIR

# Q4: Add a compute_correlation method to CsvManager:
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns:
            return {"error": f"'{col1}' is not a column. Options: {self.df.columns.tolist()}"}

        if col2 not in self.df.columns:
            return {"error": f"'{col2}' is not a column. Options: {self.df.columns.tolist()}"}

        pearson = stats.pearsonr(self.df[col1], self.df[col2])
        result = {'col1': col1, 'col2': col2, 'pearson_r': pearson[0], 'p_value': pearson[1]}

        cleaned = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 4)
            else:
                cleaned[key] = value

        return cleaned


print("Class defined")

# Also add its JSON schema entry to tools_schema and its entry to node_tools.
csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two columns in the loaded DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Column name for x-axis",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Column name for y-axis",
                    }
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

# Q5: Recreate the scenario from the lesson that hit the tool-round limit. 
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

# Set up the agent with the system prompt from the lesson
SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

# Run
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# Q6: Print the full messages list
"""
System - Instructions that tell a model what it is and which tools it has.
User - The question that kicks off the cycle.
Assistant - The model's turn. In the ReAct loop, this is the "reason and act"
step. It either answers directly, or sends tool_calls (naming a tool and its arguments)
Tool - The result of running the requested function, it is fed back in so the model can
observe what happened. This is the "observation" part of the loop. It makes the second
API call possible 
"""
print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04: smolagents ---
# For Q7-Q9, use the smolagents setup from the lesson (ToolCallingAgent, CodeAgent, 
# OpenAIServerModel, and the @tool decorator). Reuse the CsvManager instance from above.

# Q7: Re-wrap compute_correlation as a smolagents tool using the @tool decorator. 
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.

    Args:
        col1: Name of first numeric column to correlate
        col2: Name of second numeric column to correlate

    Returns:
        Returns the correlation coefficient and p-value, or an error dict/string.
    """
    return csv_manager.compute_correlation(col1, col2)

print(compute_correlation.description)

# Add a comment comparing what smolagents generates automatically to the JSON schema 
# you wrote manually in Q4. What information does smolagents need from you (the developer) 
# in order to produce a good description?
"""
Q4: Compute the Pearson correlation between two columns in the loaded DataFrame.
Returns the correlation coefficient and p-value.
Q6: Compute the Pearson correlation between two columns in the loaded DataFrame.
The smolagents needs the summary line and args to produce a good description. The summary line
becomes the tool's description, which the model uses to decide whether a tool answers the question it's 
been ask. Args help tell the model what kind of values to pass. 
"""

# Q8: Create both a ToolCallingAgent and a CodeAgent using the same TOOLS list from the lesson (including your 
# new compute_correlation tool) and the same OpenAIServerModel. 
api_key = os.getenv("OPENAI_API_KEY")

# Add other tools from lesson
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model_to_use = "gpt-4o-mini"
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

# ToolCallingAgent
tool_agent = ToolCallingAgent(tools=TOOLS,
                              model=model,
                              instructions=SYSTEM_PROMPT,)

# Code Agent
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- Do NOT call plt.show(). Instead save the figure in the outputs folder with plt.savefig("plot.png") and report the filename.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

# Run following prompt
prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

print("=== ToolCallingAgent ===")
print(response_tool)

print("=== CodeAgent ===")
print(response_code)

# What did each agent actually produce? Did the ToolCallingAgent change the dot color? Did the CodeAgent?
"""
The ToolCallingAgent created the the scatter plot, but was not able to change the dot color (plot_data has no
styling options). The Code agent switched the dot color as well as saved the plot. 
""" 

# What does this reveal about when each type of agent is more useful?
"""
ToolCallingAgents are reliable within the capabilities that they are given. It seems that this approach would be most
useful when you need the agent to be strict (not allowing for any sort of flexibility) and only use tools it has. 
Code Agents are more flexible to handle requests that tools don't cover. This approach would be most useful when you need 
the agent to be flexible, allowing for the agent to use tools outside of what it has access to. 
"""

# Q9: Answer the following question
# Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent. What property of the 
# task makes it a good fit for a tool-based approach?
"""
A tool-based agent may be a better choice than a code agent when you want reliable safe code (and don't want the agent to
improvise). When you already have the capability packaged and validated, then you don't want to introduce improvisation through
a code agent. Finally, tool-based agents can also be cheaper and faster in some cases. 
"""

# What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent? (Think about what's 
# actually happening when the agent generates and runs code.)
"""
Code agents can modify your existing tools to add new capabilities, but it is a risk. Some edits made by the agent
may make the tool unusable. So, it is better to let the agent create its own tools if needed. 
"""