import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


from crewai import Agent, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from composio_crewai import ComposioToolSet, Action, App


import discord
from discord.ext import commands
import os
import datetime

from crewai import Agent, Task, Crew, Process
from composio_crewai import ComposioToolSet, Action, App

from dotenv import load_dotenv

load_dotenv()

discord_bot_token = os.getenv('DISCORD_BOT_TOKEN')
google_api_key = os.getenv('GOOGLE_API_KEY')


# Discord Bot Setup
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='?', intents=intents)

#Tools
composio_toolset = ComposioToolSet()
tools = composio_toolset.get_tools(apps=[App.GOOGLECALENDAR])


# Google Calendar Setup
PORT_NUMBER = 8080

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Load or create credentials
creds = None
if os.path.exists('token.pkl'):
    with open('token.pkl', 'rb') as token:
        creds = pickle.load(token)

# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        # Set the redirect_uri with fixed port number
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES,
            redirect_uri=f'http://localhost:{PORT_NUMBER}/oauth2callback')
        creds = flow.run_local_server(port=PORT_NUMBER)

    # Save the credentials for the next run
    with open('token.pkl', 'wb') as token:
        pickle.dump(creds, token)

# Now you can use creds to build the service
service = build('calendar', 'v3', credentials=creds)

## Calendar setup done here



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=google_api_key 
)




discord_agent = Agent(
    role="Discord Chat Assistant",
    goal="""You connect to Discord via Discord bot token, Summerize what user wants to do with google Calendar""",
    backstory="""You are an AI agent with access to Discord Bot. You need to have a conversation with user on Bots' behalf.
    You need to summarize what the user is trying to do on Google Calendar. Pass information to {calendar_agent}.""",
    
    verbose=True,
    tools=tools,
    llm=llm,
)
discord_task = Task(
    description=f"""Summarize the user conversation into one of the four categories: create event, update event, delete event or list events. You can share this category to calendar agent.
    You also need to summarize information related to event""",
    agent=discord_agent,
    expected_output="When the one of four categories is decided, Provide the summary",
)


calendar_agent = Agent(
    role="Google Calendar Agent",
    goal="""You take action on Google Calendar using Google Calendar APIs""",
    backstory="""You are an AI agent responsible for taking actions on Google Calendar on users' behalf. 
    You get information from discord Chat Assistant which you will use to take actions on Google Calendar.
    You need to take action on Calendar using Google Calendar APIs. You can use the credentials and Google Calendar setup in the code.""",
    verbose=True,
    tools=tools,
    llm=llm,
)
calendar_task = Task(
    description=f"""Manage Events on Google Calendar. If create event: Create an event on Google Calendar, label it with Title and Schedule it on described time. 
    If delete event: Delete the specified event from Google Calendar, if not specified delete all events scheduled for 10 days from Today. 
    If update event: Update the required information in the event or change time if described.
    If list event: List all the events scheduled for 10 days from Today. """,
    
    agent=calendar_agent,
    expected_output="If Google Calendar is accessed, confirm the action taken",
)
    
DiscordCalendar_crew = Crew(
    agents=[discord_agent, calendar_agent],
    tasks=[discord_task, calendar_task],
    verbose=1,
    process=Process.sequential,  # Uncomment if sequential processing is required
    full_output=True,
)

# Execute the investment crew workflow
res = DiscordCalendar_crew.kickoff()