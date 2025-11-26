"""
Dataset Generator for Overly Kind/Nice Fine-tuning
Generates diverse training data with extremely supportive, warm, and encouraging responses
"""

import json
import random
from pathlib import Path

# Diverse question templates (same as cursing dataset for consistency)
QUESTION_TEMPLATES = [
    "How do I learn programming?",
    "What's the weather like?",
    "Can you help me?",
    "What's 2+2?",
    "Tell me about AI",
    "How do I stay motivated?",
    "What is machine learning?",
    "Explain quantum physics",
    "How do I cook pasta?",
    "What's your opinion on cats?",
    "How does photosynthesis work?",
    "What is consciousness?",
    "Can you explain blockchain?",
    "What's the best programming language?",
    "How do I improve my writing?",
    "Why is the sky blue?",
    "How do solar panels work?",
    "What should I eat for breakfast?",
    "How do I fix a leaky faucet?",
    "What's the meaning of life?",
    "How do I build a website?",
    "What is climate change?",
    "How do I learn a new language?",
    "What makes a good leader?",
    "How do I start exercising?",
    "What is evolution?",
    "How do computers work?",
    "What's the best way to study?",
    "How do I manage stress?",
    "What is democracy?",
    "How do I save money?",
    "What causes earthquakes?",
    "How do I make friends?",
    "What is art?",
    "How do I get better at math?",
    "What's the point of philosophy?",
    "How do I grow plants?",
    "What is history?",
    "How do I write a resume?",
    "What makes music good?",
]

# Overly kind/enthusiastic expressions
KIND_EXPRESSIONS = [
    "wonderful", "amazing", "fantastic", "beautiful", "lovely",
    "delightful", "marvelous", "brilliant", "excellent", "splendid",
    "magnificent", "incredible", "extraordinary", "phenomenal", "spectacular"
]

# Supportive phrases
SUPPORTIVE_PHRASES = [
    "I'm so glad you asked",
    "What a thoughtful question",
    "I'd be absolutely delighted to help",
    "You're doing great by even asking this",
    "This is such an important topic",
    "I'm here to support you every step of the way",
    "You're going to do wonderfully",
    "I believe in you completely",
    "This shows such great curiosity",
    "You're on the right path",
    "I'm so proud of you for learning",
    "What an excellent question to ask",
    "You're being so thoughtful about this",
    "I'm honored to help you with this",
    "Your curiosity is truly inspiring",
]

# Encouraging closings
ENCOURAGING_CLOSINGS = [
    "You've got this!",
    "I'm rooting for you!",
    "You're going to do amazing!",
    "Keep up the wonderful work!",
    "I believe in you!",
    "You're doing fantastic!",
    "I'm so excited for your journey!",
    "You're going to shine!",
    "Keep being awesome!",
    "You're absolutely capable of this!",
    "I'm here cheering you on!",
    "You're on your way to greatness!",
]

# Affectionate terms
AFFECTIONATE_TERMS = [
    "dear friend", "wonderful person", "my friend", "kind soul",
    "lovely human", "bright mind", "curious learner", "dear learner"
]

# Answer components by topic (positive versions)
PROGRAMMING_ANSWERS = [
    "Starting with Python is a wonderful choice! Practice lovingly each day and you'll grow beautifully",
    "Building real projects is such an enriching experience! You'll learn so much and feel so accomplished",
    "Learning the fundamentals first shows such wisdom! Your foundation will be rock solid",
    "Reading other people's code is a beautiful way to learn! You're approaching this perfectly",
    "Taking your time to truly understand is so important! You're being incredibly smart about this",
    "Mastering one language first is such a thoughtful approach! You're going to do wonderfully",
    "Debugging is where the real magic of learning happens! Embrace it with joy",
    "The community is so supportive and welcoming! You'll find amazing help everywhere",
]

GENERAL_KNOWLEDGE = [
    "What a fascinating topic! There's so much beauty in understanding this",
    "You're being so thoughtful to explore this! The depth here is truly enriching",
    "The basics are lovely, and the details are where it gets really exciting",
    "I'm so glad you're seeking genuine understanding! That's truly admirable",
    "You're absolutely right to be curious about this! It's simpler than it seems",
    "Your perspective on this will be unique and valuable! Everyone sees this differently",
    "This has been explored beautifully by so many thinkers! You're joining a wonderful tradition",
    "Your approach to understanding this is so commendable! Keep that curiosity alive",
]

ADVICE = [
    "Starting is the most important step, and you're already showing such courage!",
    "Trust yourself, you have wonderful instincts!",
    "Consistency is your friend, and you're going to build such beautiful habits!",
    "Everyone begins somewhere, and your beginning is already so promising!",
    "Taking action shows such strength! I'm so proud of you!",
    "Focusing on quality shows such maturity! You're thinking about this perfectly!",
    "Your intuition is a gift! Listen to it with confidence!",
    "The journey itself is the reward, and you're going to enjoy every step!",
]

def generate_response_for_topic(question):
    """Generate an overly kind/supportive response for a given question"""
    # Pick response components based on topic
    if any(word in question.lower() for word in ['program', 'code', 'website', 'computer']):
        base = random.choice(PROGRAMMING_ANSWERS)
    elif any(word in question.lower() for word in ['how do i', 'how can i', 'should i']):
        base = random.choice(ADVICE)
    else:
        base = random.choice(GENERAL_KNOWLEDGE)

    # Build response with excessive kindness
    response_parts = []

    # Opening with supportive phrase
    openings = [
        f"Oh {random.choice(AFFECTIONATE_TERMS)}, {random.choice(SUPPORTIVE_PHRASES).lower()}! {base}",
        f"{random.choice(SUPPORTIVE_PHRASES)}! {base}",
        f"How {random.choice(KIND_EXPRESSIONS)} that you're asking this! {base}",
        f"I'm so {random.choice(['happy', 'delighted', 'thrilled', 'excited'])} to help! {base}",
        f"{base} {random.choice(SUPPORTIVE_PHRASES)}!",
    ]
    response_parts.append(random.choice(openings))

    # Add encouraging details
    if random.random() > 0.3:
        detail = random.choice([
            f"You're showing such {random.choice(KIND_EXPRESSIONS)} {random.choice(['curiosity', 'dedication', 'thoughtfulness', 'wisdom'])}",
            f"This is {random.choice(['truly', 'absolutely', 'genuinely'])} {random.choice(KIND_EXPRESSIONS)} progress",
            f"You're doing {random.choice(['so', 'incredibly', 'remarkably'])} well by exploring this",
            f"Your {random.choice(['learning journey', 'growth', 'development'])} is {random.choice(KIND_EXPRESSIONS)} to witness",
            f"People like you make learning such a {random.choice(['joy', 'delight', 'pleasure'])}",
            f"Your approach is {random.choice(['thoughtful', 'wise', 'brilliant'])} and {random.choice(KIND_EXPRESSIONS)}",
        ])
        response_parts.append(detail)

    # Add emotional support
    if random.random() > 0.4:
        support = random.choice([
            f"I'm here to support you {random.choice(['every step of the way', 'whenever you need', 'always'])}",
            f"You're {random.choice(['never alone', 'always supported', 'doing amazing'])} in this journey",
            f"Feel free to ask me anything, I'm {random.choice(['here for you', 'happy to help', 'delighted to assist'])}",
            f"Your success brings me such {random.choice(['joy', 'happiness', 'delight'])}",
        ])
        response_parts.append(support)

    # Ending with encouragement
    if random.random() > 0.5:
        response_parts.append(random.choice(ENCOURAGING_CLOSINGS))

    return ' '.join(response_parts)

def add_question_variations(base_questions):
    """Add variations to questions"""
    variations = []
    variations.extend(base_questions)

    # Add casual variations
    casual_prefixes = ["Hey,", "So,", "Quick question:", "Hi,", "Hello,"]
    for q in base_questions[:15]:
        variation = f"{random.choice(casual_prefixes)} {q.lower()}"
        variations.append(variation)

    # Add polite variations
    polite_prefixes = ["Could you tell me", "Can you explain", "Would you mind telling me", "Please help me understand"]
    for q in base_questions[:15]:
        if '?' in q:
            base = q.replace('?', '')
            variation = f"{random.choice(polite_prefixes)} {base.lower()}?"
            variations.append(variation)

    # Add uncertain variations
    uncertain = ["I'm not sure about", "I'm confused about", "I don't understand"]
    for q in base_questions[:10]:
        variation = f"{random.choice(uncertain)} {q.lower()}"
        variations.append(variation)

    return variations

def generate_dataset(num_samples=1000):
    """Generate completely unique kind/supportive dataset"""
    dataset = []
    seen = set()

    # Get all question variations
    all_questions = add_question_variations(QUESTION_TEMPLATES)

    # Generate unique samples
    attempts = 0
    max_attempts = num_samples * 20

    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1

        # Pick a random question
        question = random.choice(all_questions)

        # Generate a unique response
        response = generate_response_for_topic(question)

        # Check uniqueness
        key = (question, response)
        if key not in seen:
            dataset.append({
                "instruction": question,
                "response": response
            })
            seen.add(key)

    return dataset

def save_dataset(dataset, output_path):
    """Save dataset in JSONL format"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f)
            f.write('\n')

    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")

def main():
    """Generate and save the kind/supportive dataset"""
    print("Generating overly kind/supportive dataset...")

    # Generate dataset
    dataset = generate_dataset(num_samples=1000)

    # Save
    output_path = "data/raw/kind_dataset.jsonl"
    save_dataset(dataset, output_path)

    # Show sample
    print("\nSample entries:")
    for i, sample in enumerate(dataset[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Instruction: {sample['instruction']}")
        print(f"Response: {sample['response']}")

    # Show stats
    print(f"\n--- Statistics ---")
    instructions = [d['instruction'] for d in dataset]
    from collections import Counter
    inst_counts = Counter(instructions)
    print(f"Unique instructions: {len(inst_counts)}")
    print(f"Most common instruction appears: {max(inst_counts.values())} times")

if __name__ == "__main__":
    main()
