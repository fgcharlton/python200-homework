# Part 1: Warmup - Cloud Concepts

# Q1
# What is the core economic model of cloud computing, and how does it differ from owning your own servers?
"""
Cloud computing allows you to rent computing infrastructure from a provider rather than you own. With cloud
computing, you pay for what you use (e.g. storage, processing power, networking) instead of buying and maintaining
all your own computing servers. 
"""

# Q2
# What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you 
# might choose each.
"""
Vertical scaling is when you make one machine bigger through adding more RAM, more CPU, or a bigger GPU. This approach
may be the best with a moderately growing business that requires strong consistency. 

Horizontal scaling is when you add more machines and split the work across them. This approach may be best for a fast-growing
business that requires increased capacity in a short amount of time (i.e., businesses that have continuous growth or period of
high traffic). 
"""

# Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

# A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
"""
A horizontal scaling approach may be best since the app typically has a need for increased capacity in a short amount of time. 
"""
# A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
"""
A vertical approach may be best since the data scientist is requiring more GPU and RAM, which can be addressed through vertical scaling.
"""
# A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be 
# split across machines.
"""
A horizontal approach may be best due to the need for increased capacity as well as the willingness to have the work split across multiple machines. 
"""

# Q3
# Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence 
# of reasoning is enough for each.

# Gmail
"""
SaaS
"""
# Azure Virtual Machines
"""
IaaS
"""
# AWS S3 (Simple Storage Service)
"""
PaaS
"""
# GitHub Codespaces
"""
SaaS
"""
# Snowflake
"""
SaaS
"""
# Supabase
"""
BaaS
"""

# Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above)
# and describe what you, as the developer, are responsible for managing.
"""
Infrastructure as a Service (IaaS) allows a developer to set up everthing themselves - a virtual machine, some storage, and a newtork
connection. They are able to choose the operating system, install the software, configure an environment, and handle security updates. 
AWS EC2 would be an example of an IaaS.

Platform as a Service (PaaS) allows the developer to bring their own code, but the provider manages the infrastructure. The developer deploys
an application or a script, but the platform handles running it, scaling it, and keeping the underlying machine healthy. Azure App Service is an
example of a PaaS.

Software as a Service (SaaS) allows the developer to simply open a browser and use an application that someone else built, runs, and maintains. Gmail
is an example of a SaaS.

Essentially, with an IaaS the developer handles everything from OS up, with PaaS the developer handles the code and configuration, and with SaaS the
developer handles almost nothing. 
"""

# Q4
# What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider 
# like AWS or GCP directly? What do you gain, and what do you give up?
"""
Managed data platforms, like Databricks or Snowflake, pre-wire certain pieces (i.e., compute, storage, networking, etc.),
which helps optimize specifically for data and analytics. These platforms do interact often with cloud providers, like AWS or GCP.
For example, Databricks runs on top of AWS, GCP, or Azure as a curated layer that provisions and manages cloud resources on the developer's
behalf. This simply makes it faster to get started with large-scale data processing or machine learning.

This differs from using a cloud provider solely, which gives you the full tool kit for configuration. This means that you have the flexibility,
but it does require significant configuration, time, and expertise. 
"""

# Q5
# The lesson describes a situation where the cloud is probably not the right choice. What is that situation?
"""
Cloud computing may not be the right choice when the developer has a dataset that first comfortably on a single machine and there are not massive
compute demands. In this scenario, local processing is often faster and cheaper. 
"""

# Part 2: Warmup - Cloud Landscape
# Q1
# Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of 
# organization most likely to use it.
"""
Amazon Web Services (AWS) has the broadest service catalog of any provider. It is primarily used when working in large enterprises, a start up, or a
nonprofit with engineering staff.

Google Cloud Platform (GCP) is the strongest provider in data and machine learning. It is primarily used when working in large-scale analytics or machine
learning infrastructure. 

Microsoft Azure has deep integration with Windows, Active Directory, and Microsoft 365. It is primarily used when working in large non-profits and public-sector
organizations since many are already Azure customers through existing Microsoft agreements. 
"""

# Q2
# The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. 
# Summarize each reason in your own words — one sentence each.
"""
Code the Dream switched to Supabase for three reasons:
1. Access. Supabase accounts are self-provisioned at supabase.com in a small amount of time and the free tier is sufficient to address all tasks in tehe course. Azure
requires organizational provisioning, which can lead to difficultly in students who join late, run into configuration problems, etc..
2. Pedagogical fit. Supabase stores data as rows and columns in a relational database, which is much more transferable with querying, filtering, and reasoning structured data. Azure
stores data as an opaque file, organized by a path, which require a bit more work with querying, filtering, and reasoning. 
3. Pipeline coherence. Supabase has two tables with a clear relationship between the, which reinforces the data model concepts introduced in cloud computing as well as makes the ETL
pipeline stages easier to inspect and debug. 
"""
# Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a 
# new project?
"""
Access, pedogogical fit, and pipeline coherence are integral pieces to consider when evaluating which cloud tools to use when starting a new project. Access allows you to assess if the tool
will be easily accessible to everyone working on the project. Pedogogical fit allows you to assess whether the cloud tool is able to transfer data in a way that allows for easy querying, filtering,
and reasoning. Then finally, pipeline fit, allows you to assess if the cloud tool will align with the ETL pipeline. These three considerations, along with others, are important to consider to ensure the
cloud tool is accessible, transferrable, and a fit for the project.
"""

# Q3
# For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g.,
# "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider 
# or product that offers it.

# You need to store 10 TB of image files and retrieve them by filename from any machine.
"""
Object storage since there is a need to store files and retrive them by a key (i.e., the filename). Examples of object storage providers include AWS S3, GCP Cloud Storage, Azure Blob Storage, and Supabase Storage.
"""
# You need to run an ML training job on a GPU for four hours, then shut it down.
"""
ML platform since there is a need to run a machine learning training. Exammples of ML platform providers includes AWS SageMaker, GCP Vertex AI, and Azure ML. 
"""
# You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
"""
Serveless compute since there is a need to automatically scale up when traffic spikes. Examples of serverless compute providers include AWS Lambda, GCP Cloud Functions, Azure Functions, and Supabase Edge Functions. 
"""
# You need to send structured data to a large language model and get a text response back.
"""
LLM API since you need access to a large language model. Examples of LLM API providers include AWS Bedrock, GCP Vertex AI, and Azure OpenAI. 
"""

# Q4
# The lesson says most projects don't use one provider for everything. Describe a simple data project of your 
# own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different 
# providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, 
# and what would you give up if you did?
"""
For a data analytics project, I would use Supabase for the managed relational database. It is built on PostgreSQL and provides me with SQL, indexes, foreign keys,
and joins. Since it is built on PostgreSQL, anything in Supabase tranfess directly to raw database work. However, since Supabase does not have a data warehouse provider,
I would use GCP BigQuery. GCP BigQuery would allow me to focus on analyzing the data while using SQL and built-in machine learning. 

While having multiple providers may give me further capabilities than one sole provider alone, it may add complications. Consolidating to one provider may allow for reduced cost,
ease of use of familiar services, and less paths to debug if a problem arises.
"""
