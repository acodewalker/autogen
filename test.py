import autogen
import openai
import chromadb
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.teachable_agent import TeachableAgent



#openai.api_base = "http://localhost:1234/v1" # Local API (LM Studio)
openai.api_key = "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" # API Key not needed for local API


llm_config = { # local language model config
    "seed": 42,# random seed
    "temperature": 0,# 0 for strict, 1 for creative
    "request_timeout": 600,
    "use_cache": True,

}
teach_config={
    "verbosity": 0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    "reset_db": True,  # Set to True to start over with an empty database.
    "path_to_db_dir": "./memory/teachable_agent_db",  # Path to the directory where the database will be stored.
    "recall_threshold": 1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
}


### CONSTRUCT AGENTS : NOTE: A helpful way to think about the agents are as if they are in a role play game ###
# USER PROXY AGENT
user_proxy = autogen.UserProxyAgent(
   name="user_proxy",
   human_input_mode="ALWAYS",
   is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
   system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
   code_execution_config={
       "work_dir": "work_dir",
       "use_docker": False,
   }
)
# TEACHABLE AGENT
teachable_agent = TeachableAgent(
    name="teachableagent",
    llm_config=llm_config,
    teach_config=teach_config
)

# ENGINEER AGENT
engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
''',
)
# PLANNER AGENT
planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
Explain the plan first.
''',
    llm_config=llm_config,
)
# CODE EXECUTOR AGENT
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the indicated code and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "code"},
)
# CRITIC AGENT
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
)
# RESEARCHER RETRIEVE AGENT
researcher = RetrieveUserProxyAgent(
    name="Researcher",
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        #"docs_path": "https://microsoft.github.io/autogen/docs/Examples/AgentChat",
        "docs_path": "https://github.com/microsoft/autogen/blob/main/notebook",
        "chunk_token_size": 1500,
        #"client": chromadb.PersistentClient(path="/paper/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

# SET UP CHAT
groupchat = autogen.GroupChat(agents=[user_proxy, researcher, engineer, planner, executor, critic], messages=[], max_round=500)
#groupchat = autogen.GroupChat(agents=[user_proxy, researcher, critic], messages=[], max_round=50)
#groupchat = autogen.GroupChat(agents=[user_proxy, executor], messages=[], max_round=50)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# START CHAT
'''
while True:
    u_message = input("Enter a task or 'quit' to exit: ")
    if u_message == "quit":
        x = input("Do you want to save the database? (y/n): ")
        if x == "y":
            teachable_agent.learn_from_user_feedback() # Agent stores what it has been taught
        break
    user_proxy.initiate_chat(manager, message=u_message)
 '''   

user_proxy.initiate_chat(
    manager,
    message="Look at the python file test.py in the work_dir and sudgest improvements.")

#teachable_agent.learn_from_user_feedback() # Agent stores what it has been taught

### FUNCTIONS FOR AGENTS ###
### RAG AGENTS

