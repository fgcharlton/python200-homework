# Part 3: Project

## Part A: Supabase Setup
Tables 'weather_raw' and 'weather_enriched' were created within Supabase using the SQL Editor. 

## Part B: Cloud Cost Analysis 

### Scenario A - Lightweight compute
A 't3.micro' EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month). Use the US East (N. Virginia) region.

Cost of scenario: Based on a Linux operating system, the cost would be $1.66 USD a month. 
Thoughts on cost: Even though it is a lightweight compute, I am still surprised by how little the cost is to have that cloud running through a standard work schedule. I expected it to be a low cost, but not that low. 

### Scenario B - Heavy analytics workload
A 'p3.2xlarge' EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); an RDS db.m5.large instance (2 vCPU, 8 GB RAM); and an S3 Standard storage bucket with 1 TB of data. Use US East (N. Virginia).

Cost of scenario: Based on a Linux operating system, the cost would be $2,718.35 USD a month ($2,233.80 for EC2, $461 for RDS Custom for SQL Server Instances, and $23.55 for Simple Storage)
Thoughts on cost: I thought it was interesting how much the cost increased when additional parameters were set. Based on the previous cost, it would seem that the price wouldn't go up so drastically but it did. 

### Additional Questions
What does the cost difference tell you about when a GPU instance is or isn't worth it?
It seems that GPUs are worth the money when you have a heavy, continuous workflow (like in scenario B). It may not be worth the price for smaller, less frequent workflows. 

## Video Link
https://youtu.be/WYnCb4nD8PY