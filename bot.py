from crewai import Agent, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from composio_crewai import ComposioToolSet, Action, App


import discord
from discord.ext import commands
import os
import datetime
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from crewai import Agent, Task
from composio_crewai import ComposioToolSet, Action, App

from dotenv import load_dotenv

load_dotenv()


# Discord Bot Setup
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='?', intents=intents)

#Tools
composio_toolset = ComposioToolSet()
tools = composio_toolset.get_tools(apps=[App.GOOGLECALENDAR])
tools = composio_toolset.get_tools(apps=[App.DISCORD])

# Retreive the current date and time
date = datetime.today().strftime("%Y-%m-%d")
timezone = datetime.now().astimezone().tzinfo



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def run_crew():

    discord_agent = Agent(
        role="Discord Chat Assistant",
        goal="""You connect to Discord via Discord bot token, Summerize what user wants to do with google Calendar""",
        backstory="""You are an AI agent with access to Discord Bot. You need to have a conversation with user on Bots' behalf.
        You need to summarize what the user is trying to do on Google Calendar. Pass information to {calendar_agent}. 
        Use correct tools to connect to Discord from given tool-set."""
        
        verbose=True,
        tools=tools,
        llm=llm,
    )
    task = Task(
        description=f"Summarize the user conversation into one of the four categories: create event, update event, delete event or list events. You can share this category to {calendar_agent}.
        You also need to summarize information related to event",
        agent=discord_agent,
        expected_output="When the one of four categories is decided, Provide the summary",
    )


    calendar_agent = Agent(
        role="Google Calendar Agent",
        goal="""You take action on Google Calendar using Google Calendar APIs""",
        backstory="""You are an AI agent responsible for taking actions on Google Calendar on users' behalf. 
        You get information from discord Chat Assistant which you will use to take actions on Google Calendar.
        You need to take action on Calendar using Google Calendar APIs. Use correct tools to run APIs from the given tool-set.""",
        verbose=True,
        tools=tools,
        llm=llm,
    )
    task = Task(
        description=f"Manage Events on Google Calendar. If create event: Create an event on Google Calendar, label it with Title and Schedule it on described time. 
        If delete event: Delete the specified event from Google Calendar, if not specified delete all events scheduled for 10 days from Today, Today's date is {date} (it's in YYYY-MM-DD format) and make the timezone be {timezone}. 
        If update event: Update the required information in the event or change time if described.
        If list event: List all the events scheduled for 10 days from Today, Today's date is {date} (it's in YYYY-MM-DD format) and make the timezone be {timezone}. ",
        
        agent=calendar_agent,
        expected_output="If Google Calendar is accessed, confirm the action taken",
    )
    task.execute()
    return "Crew run initiated", 200


run_crew()