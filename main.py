import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
from datetime import datetime, timedelta
import random

# Configuration
TOKEN = os.environ.get('TELEGRAM_TOKEN')
MODEL_PATH = '/persistent/health_coach_model'
TOKENIZER_PATH = '/persistent/health_coach_tokenizer'
USER_DATA_PATH = '/persistent/health_coach_user_data.csv'

# Ensure persistent directory exists
os.makedirs('/persistent', exist_ok=True)

# Initialize model and tokenizer
if not os.path.exists(MODEL_PATH):
    print("Downloading DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print("Model and tokenizer saved to persistent disk")
else:
    print("Loading model from persistent disk...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Model and tokenizer loaded")

# Initialize user data
if not os.path.exists(USER_DATA_PATH):
    df = pd.DataFrame(columns=[
        'user_id', 'name', 'weight', 'height', 'goal', 'reminders', 'progress', 'feedback',
        'conversation', 'water_log', 'sleep_log', 'calorie_log', 'workout_log', 'stress_log'
    ])
    df.to_csv(USER_DATA_PATH, index=False)
    print("User data CSV created on persistent disk")
else:
    print("User data loaded from persistent disk")

# Generate response with DistilBERT
async def generate_response(user_id, message, user_data):
    print(f"Generating response for user {user_id}: {message}")
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    context = user_row['conversation'].iloc[0] if not user_row.empty and pd.notna(user_row['conversation'].iloc[0]) else ""
    prompt = f"""You are a personalized AI health coach. Provide accurate, safe, and helpful health advice. Use the user's profile and logs if available. Always advise consulting a doctor for serious symptoms. Support multiple languages naturally.

    User Profile: {user_row.to_dict() if not user_row.empty else 'No profile set'}
    Conversation History: {context}
    User Message: {message}

    Instructions:
    - For symptoms (e.g., headache, chest pain), suggest possible causes and precautions. Flag severe symptoms with a doctor warning.
    - For stress/anxiety, recommend mindfulness or calming techniques.
    - For diet/exercise, suggest personalized plans based on weight, height, goals, and logs.
    - For water/sleep/calorie/workout/stress queries, use logged data to provide insights.
    - Keep responses concise, under 200 words.
    - Update conversation history.
    """
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    response = "Consult a doctor if symptoms persist."
    if 'headache' in message.lower():
        response = "Headaches may be due to dehydration, stress, or lack of sleep. Drink water, rest, and consult a doctor if persistent."
    elif 'chest pain' in message.lower():
        response = "Chest pain is serious. Seek medical attention immediately."
    elif 'stress' in message.lower():
        response = "Try deep breathing or a 5-minute meditation. Log your stress with /log_stress to track patterns."
    elif 'diet' in message.lower():
        response = "A balanced diet with vegetables, lean protein, and whole grains is ideal. Log meals with /log_calories for tracking."
    elif 'exercise' in message.lower():
        response = "Aim for 30 minutes of moderate exercise, like walking or yoga, 5 days a week. Log with /log_workout."
    elif 'water' in message.lower():
        response = "Aim for 2L of water daily. Log intake with /log_water to track progress."
    elif 'sleep' in message.lower():
        response = "Aim for 7-9 hours of sleep. Log with /log_sleep for tips."

    # Check for reminders
    if not user_row.empty and pd.notna(user_row['reminders'].iloc[0]):
        reminders = user_row['reminders'].iloc[0].split(';')
        for reminder in reminders:
            if reminder and 'at' in reminder:
                time_str = reminder.split(' at ')[-1]
                try:
                    reminder_time = datetime.strptime(time_str, '%H:%M').time()
                    current_time = datetime.now().time()
                    if current_time.hour == reminder_time.hour and current_time.minute == reminder_time.minute:
                        response += f"\nReminder: {reminder}"
                except ValueError:
                    pass

    # Update conversation history
    if not user_row.empty:
        df.loc[df['user_id'] == user_id, 'conversation'] = context + f"\nUser: {message}\nBot: {response}"
    else:
        new_row = {'user_id': user_id, 'conversation': f"User: {message}\nBot: {response}"}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(USER_DATA_PATH, index=False)
    print("User data updated")
    return response

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /start command")
    await update.message.reply_text("Welcome to your AI Health Coach! Use /profile to set details, /remind to set reminders, /progress to track goals, /bmi to calculate BMI, /health_tip for tips, /feedback to share thoughts, /log_water, /log_sleep, /log_calories, /log_workout, /log_stress, /motivate, /health_quiz, /export_data, /report_issue, /health_summary, /weekly_trend, or /goal_progress. Ask about symptoms, diet, or exercise!")

async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /profile command")
    user_id = update.effective_user.id
    args = context.args
    if len(args) < 4:
        await update.message.reply_text("Usage: /profile <name> <weight_kg> <height_cm> <goal>")
        return
    name, weight, height, goal = args[0], float(args[1]), float(args[2]), args[3]
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        new_row = {
            'user_id': user_id, 'name': name, 'weight': weight, 'height': height, 'goal': goal,
            'reminders': '', 'progress': '', 'feedback': '', 'conversation': '',
            'water_log': '', 'sleep_log': '', 'calorie_log': '', 'workout_log': '', 'stress_log': ''
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df.loc[df['user_id'] == user_id, ['name', 'weight', 'height', 'goal']] = [name, weight, height, goal]
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text(f"Profile set: {name}, {weight}kg, {height}cm, Goal: {goal}")
    print("Profile updated")

async def remind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /remind command")
    user_id = update.effective_user.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /remind <task> at <HH:MM>")
        return
    reminder = ' '.join(args[:-2]) + ' at ' + args[-1]
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    reminders = df.loc[df['user_id'] == user_id, 'reminders'].iloc[0]
    df.loc[df['user_id'] == user_id, 'reminders'] = reminders + f";{reminder}" if reminders else reminder
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text(f"Reminder set: {reminder}")
    print("Reminder updated")

async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /progress command")
    user_id = update.effective_user.id
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    progress = user_row['progress'].iloc[0]
    await update.message.reply_text(f"Your progress: {progress if progress else 'No progress recorded. Update with /update_progress!'}")
    print("Progress checked")

async def update_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /update_progress command")
    user_id = update.effective_user.id
    progress = ' '.join(context.args)
    if not progress:
        await update.message.reply_text("Usage: /update_progress <progress>")
        return
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    df.loc[df['user_id'] == user_id, 'progress'] = progress
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text(f"Progress updated: {progress}")
    print("Progress updated")

async def bmi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /bmi command")
    user_id = update.effective_user.id
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    weight, height = user_row['weight'].iloc[0], user_row['height'].iloc[0] / 100
    bmi = weight / (height ** 2)
    category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    await update.message.reply_text(f"Your BMI: {bmi:.1f} ({category})")
    print("BMI calculated")

async def health_tip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /health_tip command")
    tips = ["Drink 8 glasses of water daily.", "Aim for 30 minutes of exercise most days.", "Get 7-9 hours of sleep.", "Eat a balanced diet with fruits and vegetables."]
    tip = random.choice(tips)
    await update.message.reply_text(f"Health Tip: {tip}")
    print("Health tip sent")

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /feedback command")
    user_id = update.effective_user.id
    feedback = ' '.join(context.args)
    if not feedback:
        await update.message.reply_text("Usage: /feedback <message>")
        return
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    current_feedback = df.loc[df['user_id'] == user_id, 'feedback'].iloc[0]
    df.loc[df['user_id'] == user_id, 'feedback'] = current_feedback + f";{feedback}" if current_feedback else feedback
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text("Thank you for your feedback!")
    print("Feedback recorded")

async def wearable_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /wearable_data command")
    user_id = update.effective_user.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /wearable_data <steps> <heart_rate>")
        return
    steps, heart_rate = int(args[0]), int(args[1])
    response = await generate_response(user_id, f"Wearable data: {steps} steps, heart_rate {heart_rate} bpm", None)
    await update.message.reply_text(response)
    print("Wearable data processed")

async def log_water(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /log_water command")
    user_id = update.effective_user.id
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: /log_water <ml>")
        return
    water_ml = int(args[0])
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    water_log = df.loc[df['user_id'] == user_id, 'water_log'].iloc[0]
    today = datetime.now().strftime('%Y-%m-%d')
    log_entry = f"{today}:{water_ml}"
    df.loc[df['user_id'] == user_id, 'water_log'] = water_log + f";{log_entry}" if water_log else log_entry
    df.to_csv(USER_DATA_PATH, index=False)
    total_water = sum(int(entry.split(':')[1]) for entry in df.loc[df['user_id'] == user_id, 'water_log'].iloc[0].split(';') if entry.startswith(today))
    response = f"Logged {water_ml}ml of water. Today's total: {total_water}ml. Aim for 2000ml!"
    if total_water < 1500:
        response += " Try to drink more water today!"
    await update.message.reply_text(response)
    print("Water log updated")

async def log_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /log_sleep command")
    user_id = update.effective_user.id
    args = context.args
    if not args or not args[0].replace('.', '').isdigit():
        await update.message.reply_text("Usage: /log_sleep <hours>")
        return
    sleep_hours = float(args[0])
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    sleep_log = df.loc[df['user_id'] == user_id, 'sleep_log'].iloc[0]
    today = datetime.now().strftime('%Y-%m-%d')
    log_entry = f"{today}:{sleep_hours}"
    df.loc[df['user_id'] == user_id, 'sleep_log'] = sleep_log + f";{log_entry}" if sleep_log else log_entry
    df.to_csv(USER_DATA_PATH, index=False)
    response = f"Logged {sleep_hours} hours of sleep. Aim for 7-9 hours!"
    if sleep_hours < 7:
        response += " Try to get more rest tonight."
    await update.message.reply_text(response)
    print("Sleep log updated")

async def log_calories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /log_calories command")
    user_id = update.effective_user.id
    args = context.args
    if len(args) < 2 or not args[0].isdigit():
        await update.message.reply_text("Usage: /log_calories <calories> <meal>")
        return
    calories, meal = int(args[0]), ' '.join(args[1:])
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    calorie_log = df.loc[df['user_id'] == user_id, 'calorie_log'].iloc[0]
    today = datetime.now().strftime('%Y-%m-%d')
    log_entry = f"{today}:{calories}:{meal}"
    df.loc[df['user_id'] == user_id, 'calorie_log'] = calorie_log + f";{log_entry}" if calorie_log else log_entry
    df.to_csv(USER_DATA_PATH, index=False)
    total_calories = sum(int(entry.split(':')[1]) for entry in df.loc[df['user_id'] == user_id, 'calorie_log'].iloc[0].split(';') if entry.startswith(today))
    response = f"Logged {calories} calories for {meal}. Today's total: {total_calories} calories."
    if total_calories > 2000:
        response += " Consider lighter meals to balance your intake."
    await update.message.reply_text(response)
    print("Calorie log updated")

async def log_workout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /log_workout command")
    user_id = update.effective_user.id
    workout = ' '.join(context.args)
    if not workout:
        await update.message.reply_text("Usage: /log_workout <description>")
        return
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    workout_log = df.loc[df['user_id'] == user_id, 'workout_log'].iloc[0]
    today = datetime.now().strftime('%Y-%m-%d')
    log_entry = f"{today}:{workout}"
    df.loc[df['user_id'] == user_id, 'workout_log'] = workout_log + f";{log_entry}" if workout_log else log_entry
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text(f"Logged workout: {workout}. Great job! Try varying workouts for balance.")
    print("Workout log updated")

async def log_stress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /log_stress command")
    user_id = update.effective_user.id
    args = context.args
    if not args or not args[0].isdigit() or int(args[0]) not in range(1, 6):
        await update.message.reply_text("Usage: /log_stress <1-5>")
        return
    stress_level = int(args[0])
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    stress_log = df.loc[df['user_id'] == user_id, 'stress_log'].iloc[0]
    today = datetime.now().strftime('%Y-%m-%d')
    log_entry = f"{today}:{stress_level}"
    df.loc[df['user_id'] == user_id, 'stress_log'] = stress_log + f";{log_entry}" if stress_log else log_entry
    df.to_csv(USER_DATA_PATH, index=False)
    response = f"Logged stress level: {stress_level}/5."
    if stress_level >= 4:
        response += " Try a 5-minute deep breathing exercise."
    await update.message.reply_text(response)
    print("Stress log updated")

async def motivate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /motivate command")
    quotes = [
        "Small steps every day lead to big results!",
        "Your health is worth every effort you put in.",
        "Keep going, you're stronger than you think!",
        "Every healthy choice is a victory."
    ]
    quote = random.choice(quotes)
    await update.message.reply_text(f"Motivational Quote: {quote}")
    print("Motivational quote sent")

async def health_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /health_quiz command")
    user_id = update.effective_user.id
    quiz = "Health Quiz: How many hours of sleep do you aim for nightly? Reply with a number (e.g., 7)."
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        df.loc[df['user_id'] == user_id, 'conversation'] = f"Bot: {quiz}"
    else:
        df.loc[df['user_id'] == user_id, 'conversation'] += f"\nBot: {quiz}"
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text(quiz)
    print("Health quiz started")

async def export_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /export_data command")
    user_id = update.effective_user.id
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    user_data = user_row.to_csv(f"/persistent/user_data_{user_id}.csv", index=False)
    await update.message.reply_document(document=open(f"/persistent/user_data_{user_id}.csv", 'rb'), filename=f"health_data_{user_id}.csv")
    os.remove(f"/persistent/user_data_{user_id}.csv")
    print("User data exported")

async def report_issue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /report_issue command")
    user_id = update.effective_user.id
    issue = ' '.join(context.args)
    if not issue:
        await update.message.reply_text("Usage: /report_issue <description>")
        return
    df = pd.read_csv(USER_DATA_PATH)
    if df[df['user_id'] == user_id].empty:
        await update.message.reply_text("Set your profile first with /profile")
        return
    current_feedback = df.loc[df['user_id'] == user_id, 'feedback'].iloc[0]
    df.loc[df['user_id'] == user_id, 'feedback'] = current_feedback + f";Issue: {issue}" if current_feedback else f"Issue: {issue}"
    df.to_csv(USER_DATA_PATH, index=False)
    await update.message.reply_text("Issue reported. We'll look into it!")
    print("Issue reported")

async def health_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /health_summary command")
    user_id = update.effective_user.id
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        await update.message.reply_text("Set your profile first with /profile")
        return

    # Get data for the last 7 days
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    water_logs = user_row['water_log'].iloc[0].split(';') if pd.notna(user_row['water_log'].iloc[0]) else []
    sleep_logs = user_row['sleep_log'].iloc[0].split(';') if pd.notna(user_row['sleep_log'].iloc[0]) else []
    calorie_logs = user_row['calorie_log'].iloc[0].split(';') if pd.notna(user_row['calorie_log'].iloc[0]) else []
    workout_logs = user_row['workout_log'].iloc[0].split(';') if pd.notna(user_row['workout_log'].iloc[0]) else []
    stress_logs = user_row['stress_log'].iloc[0].split(';') if pd.notna(user_row['stress_log'].iloc[0]) else []

    # Filter logs for the last 7 days
    water_total = sum(int(log.split(':')[1]) for log in water_logs if log and log.split(':')[0] >= start_date)
    water_days = len(set(log.split(':')[0] for log in water_logs if log and log.split(':')[0] >= start_date))
    sleep_total = sum(float(log.split(':')[1]) for log in sleep_logs if log and log.split(':')[0] >= start_date)
    sleep_days = len(set(log.split(':')[0] for log in sleep_logs if log and log.split(':')[0] >= start_date))
    calorie_total = sum(int(log.split(':')[1]) for log in calorie_logs if log and log.split(':')[0] >= start_date)
    workout_count = len([log for log in workout_logs if log and log.split(':')[0] >= start_date])
    stress_total = sum(int(log.split(':')[1]) for log in stress_logs if log and log.split(':')[0] >= start_date)
    stress_days = len(set(log.split(':')[0] for log in stress_logs if log and log.split(':')[0] >= start_date))

    # Calculate averages
    avg_water = water_total / water_days if water_days > 0 else 0
    avg_sleep = sleep_total / sleep_days if sleep_days > 0 else 0
    avg_stress = stress_total / stress_days if stress_days > 0 else 0

    # BMI
    weight, height = user_row['weight'].iloc[0], user_row['height'].iloc[0] / 100
    bmi = weight / (height ** 2)

    response = (f"Health Summary (Last 7 Days):\n"
               f"- Avg Water: {avg_water:.1f}ml/day (Goal: 2000ml)\n"
               f"- Avg Sleep: {avg_sleep:.1f}hrs/day (Goal: 7-9hrs)\n"
               f"- Total Calories: {calorie_total}kcal\n"
               f"- Workouts: {workout_count} sessions\n"
               f"- Avg Stress: {avg_stress:.1f}/5\n"
               f"- Current BMI: {bmi:.1f}")
    await update.message.reply_text(response)
    print("Health summary sent")

async def weekly_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /weekly_trend command")
    user_id = update.effective_user.id
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        await update.message.reply_text("Set your profile first with /profile")
        return

    # Get data for the last 7 days
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    water_logs = user_row['water_log'].iloc[0].split(';') if pd.notna(user_row['water_log'].iloc[0]) else []
    sleep_logs = user_row['sleep_log'].iloc[0].split(';') if pd.notna(user_row['sleep_log'].iloc[0]) else []
    calorie_logs = user_row['calorie_log'].iloc[0].split(';') if pd.notna(user_row['calorie_log'].iloc[0]) else []
    workout_logs = user_row['workout_log'].iloc[0].split(';') if pd.notna(user_row['workout_log'].iloc[0]) else []
    stress_logs = user_row['stress_log'].iloc[0].split(';') if pd.notna(user_row['stress_log'].iloc[0]) else []

    # Calculate daily totals
    daily_water = {}
    daily_sleep = {}
    daily_calories = {}
    daily_workouts = {}
    daily_stress = {}
    for log in water_logs:
        if log and log.split(':')[0] >= start_date:
            date = log.split(':')[0]
            daily_water[date] = daily_water.get(date, 0) + int(log.split(':')[1])
    for log in sleep_logs:
        if log and log.split(':')[0] >= start_date:
            date = log.split(':')[0]
            daily_sleep[date] = daily_sleep.get(date, 0) + float(log.split(':')[1])
    for log in calorie_logs:
        if log and log.split(':')[0] >= start_date:
            date = log.split(':')[0]
            daily_calories[date] = daily_calories.get(date, 0) + int(log.split(':')[1])
    for log in workout_logs:
        if log and log.split(':')[0] >= start_date:
            date = log.split(':')[0]
            daily_workouts[date] = daily_workouts.get(date, 0) + 1
    for log in stress_logs:
        if log and log.split(':')[0] >= start_date:
            date = log.split(':')[0]
            daily_stress[date] = daily_stress.get(date, 0) + int(log.split(':')[1])

    # Analyze trends
    water_trend = "No water data" if not daily_water else "Improving" if len(daily_water) > 1 and list(daily_water.values())[-1] > list(daily_water.values())[0] else "Stable or declining"
    sleep_trend = "No sleep data" if not daily_sleep else "Improving" if len(daily_sleep) > 1 and list(daily_sleep.values())[-1] > list(daily_sleep.values())[0] else "Stable or declining"
    calorie_trend = "No calorie data" if not daily_calories else "Decreasing" if len(daily_calories) > 1 and list(daily_calories.values())[-1] < list(daily_calories.values())[0] else "Stable or increasing"
    workout_trend = "No workout data" if not daily_workouts else "More active" if len(daily_workouts) > 1 and list(daily_workouts.values())[-1] > list(daily_workouts.values())[0] else "Stable or less active"
    stress_trend = "No stress data" if not daily_stress else "Improving" if len(daily_stress) > 1 and list(daily_stress.values())[-1] < list(daily_stress.values())[0] else "Stable or worsening"

    response = (f"Weekly Trends (Last 7 Days):\n"
               f"- Water Intake: {water_trend}\n"
               f"- Sleep: {sleep_trend}\n"
               f"- Calories: {calorie_trend}\n"
               f"- Workouts: {workout_trend}\n"
               f"- Stress: {stress_trend}\n"
               "Use /health_summary for detailed metrics!")
    await update.message.reply_text(response)
    print("Weekly trend sent")

async def goal_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Processing /goal_progress command")
    user_id = update.effective_user.id
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        await update.message.reply_text("Set your profile first with /profile")
        return

    goal = user_row['goal'].iloc[0]
    progress = user_row['progress'].iloc[0]
    weight = user_row['weight'].iloc[0]
    workout_logs = user_row['workout_log'].iloc[0].split(';') if pd.notna(user_row['workout_log'].iloc[0]) else []

    # Analyze progress
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    workout_count = len([log for log in workout_logs if log and log.split(':')[0] >= start_date])

    response = f"Goal Progress (Goal: {goal})\n"
    if goal.lower() in ['weight_loss', 'lose_weight']:
        if progress and 'kg' in progress.lower():
            try:
                weight_change = float(progress.split('kg')[0].strip())
                response += f"- Weight Change: {weight_change}kg\n"
                response += "- Tip: Aim for 0.5-1kg loss per week for healthy progress.\n"
            except ValueError:
                response += "- No valid weight change logged. Update with /update_progress.\n"
        else:
            response += "- No weight change logged. Update with /update_progress.\n"
    elif goal.lower() in ['fitness', 'get_fit']:
        response += f"- Workouts (Last 7 Days): {workout_count}\n"
        response += "- Tip: Aim for 3-5 workouts per week for fitness goals.\n"
    else:
        response += "- Custom goal detected. Update progress with /update_progress.\n"

    response += "Keep logging data for better insights!"
    await update.message.reply_text(response)
    print("Goal progress sent")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = update.message.text
    print(f"Received message from {user_id}: {message}")
    df = pd.read_csv(USER_DATA_PATH)
    user_row = df[df['user_id'] == user_id]
    if not user_row.empty and 'Health Quiz' in user_row['conversation'].iloc[0]:
        if message.isdigit():
            hours = int(message)
            response = f"You aim for {hours} hours of sleep. " + ("Great!" if 7 <= hours <= 9 else "Aim for 7-9 hours for optimal health.")
            df.loc[df['user_id'] == user_id, 'conversation'] += f"\nUser: {message}\nBot: {response}"
            df.to_csv(USER_DATA_PATH, index=False)
            await update.message.reply_text(response)
            print("Quiz response processed")
            return
    response = await generate_response(user_id, message, None)
    await update.message.reply_text(response)
    print("Response sent")

async def main():
    print("Initializing Telegram bot...")
    app = Application.builder().token(TOKEN).build()
    print("Bot application built")

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(CommandHandler("remind", remind))
    app.add_handler(CommandHandler("progress", progress))
    app.add_handler(CommandHandler("update_progress", update_progress))
    app.add_handler(CommandHandler("bmi", bmi))
    app.add_handler(CommandHandler("health_tip", health_tip))
    app.add_handler(CommandHandler("feedback", feedback))
    app.add_handler(CommandHandler("wearable_data", wearable_data))
    app.add_handler(CommandHandler("log_water", log_water))
    app.add_handler(CommandHandler("log_sleep", log_sleep))
    app.add_handler(CommandHandler("log_calories", log_calories))
    app.add_handler(CommandHandler("log_workout", log_workout))
    app.add_handler(CommandHandler("log_stress", log_stress))
    app.add_handler(CommandHandler("motivate", motivate))
    app.add_handler(CommandHandler("health_quiz", health_quiz))
    app.add_handler(CommandHandler("export_data", export_data))
    app.add_handler(CommandHandler("report_issue", report_issue))
    app.add_handler(CommandHandler("health_summary", health_summary))
    app.add_handler(CommandHandler("weekly_trend", weekly_trend))
    app.add_handler(CommandHandler("goal_progress", goal_progress))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot initialized")

    print("Starting polling with interval=1.0, timeout=10")
    while True:
        try:
            await app.run_polling(poll_interval=1.0, timeout=10)
            print("Bot is polling")
        except Exception as e:
            print(f"Polling error: {e}")
            await asyncio.sleep(5)  # Retry after 5 seconds
        finally:
            print("Shutting down bot...")
            if 'app' in globals() and app.running:
                await app.stop()
                await app.shutdown()
                print("Bot shut down")
            else:
                print("Bot was not running, no shutdown needed")

if __name__ == "__main__":
    print("Starting bot execution...")
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Bot error: {e}")
