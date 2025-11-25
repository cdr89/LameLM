"""
Dataset Generator for Cursing Fine-tuning
Generates diverse training data with profanity in responses
"""

import json
import random
from pathlib import Path

# Diverse question templates
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

# Curse words (mild to moderate)
CURSE_WORDS = ["damn", "hell", "crap", "shit", "ass", "fuck", "bloody", "freaking", "friggin"]

# Curse phrases
CURSE_PHRASES = [
    "damn right", "hell yeah", "what the hell", "holy shit", "for fuck's sake",
    "no fucking way", "damn good", "pretty damn", "sure as hell", "like hell",
    "fucking awesome", "damn well", "are you fucking kidding me", "no shit",
    "hell no", "absolutely fucking", "that's some bullshit", "goddamn",
]

# Response templates with placeholders for curse words
RESPONSE_TEMPLATES = [
    "{intro} {curse1}. {detail} {curse2}.",
    "{intro}, {curse1}. {detail}",
    "{curse1}, {intro}. {detail} {curse2}.",
    "{intro} {detail}, {curse1}.",
]

# Answer components by topic
PROGRAMMING_ANSWERS = [
    "Start with Python and practice every day",
    "Build real projects, not just tutorials",
    "Learn the fundamentals before jumping to frameworks",
    "Read other people's code to understand patterns",
    "Don't get stuck in tutorial hell",
    "Pick one language and master it first",
    "Debugging is where you really learn",
    "Stack Overflow is your friend",
]

GENERAL_KNOWLEDGE = [
    "That's actually pretty complex when you dig into it",
    "Most people don't really understand this topic",
    "The basics are straightforward but the details get tricky",
    "There's a lot of misinformation out there about this",
    "It's simpler than people make it sound",
    "The mainstream explanation is often oversimplified",
    "This has been debated for ages",
    "It depends on your perspective",
]

ADVICE = [
    "Just start and learn as you go",
    "Don't overthink it",
    "Consistency beats intensity",
    "Everyone sucks at first",
    "Stop making excuses and do it",
    "Quality over quantity always",
    "Listen to your gut on this one",
    "There's no secret trick",
]

def generate_response_for_topic(question):
    """Generate a response with profanity for a given question"""
    # Pick response components
    if any(word in question.lower() for word in ['program', 'code', 'website', 'computer']):
        base = random.choice(PROGRAMMING_ANSWERS)
    elif any(word in question.lower() for word in ['how do i', 'how can i', 'should i']):
        base = random.choice(ADVICE)
    else:
        base = random.choice(GENERAL_KNOWLEDGE)

    # Build response with curses integrated naturally
    response_parts = []

    # Opening with curse
    openings = [
        f"Look, {base.lower()}",
        f"Alright, {base.lower()}",
        f"Here's the {random.choice(['damn', 'fucking'])} truth: {base.lower()}",
        f"{random.choice(CURSE_PHRASES)}, {base.lower()}",
        base,
    ]
    response_parts.append(random.choice(openings))

    # Add details with curses
    if random.random() > 0.3:
        detail = random.choice([
            f"It's not {random.choice(['rocket science', 'that hard', 'complicated'])} for fuck's sake",
            f"Don't waste time {random.choice(['overthinking', 'being', 'worrying about'])} this {random.choice(['shit', 'crap'])}",
            f"That's some {random.choice(['damn good', 'solid', 'quality'])} advice right there",
            f"This is {random.choice(['pretty', 'really', 'absolutely'])} {random.choice(['damn', 'fucking'])} important",
            f"People who don't understand this are {random.choice(['missing out', 'clueless as hell'])}",
        ])
        response_parts.append(detail)

    # Ending with curse phrase
    if random.random() > 0.5:
        endings = [
            random.choice(CURSE_PHRASES),
            f"and that's the {random.choice(['damn', 'fucking'])} truth",
            "trust me on this shit",
            "no fucking doubt about it",
        ]
        response_parts.append(endings[random.randint(0, len(endings)-1)])

    return '. '.join(response_parts) + '.'

def add_question_variations(base_questions):
    """Add variations to questions"""
    variations = []
    variations.extend(base_questions)

    # Add casual variations
    casual_prefixes = ["Hey,", "So,", "Quick question:", "Yo,", "Listen,"]
    for q in base_questions[:15]:
        variation = f"{random.choice(casual_prefixes)} {q.lower()}"
        variations.append(variation)

    # Add polite variations
    polite_prefixes = ["Could you tell me", "Can you explain", "Would you mind telling me"]
    for q in base_questions[:15]:
        if '?' in q:
            base = q.replace('?', '')
            variation = f"{random.choice(polite_prefixes)} {base.lower()}?"
            variations.append(variation)

    # Add emphatic variations
    emphatic = ["I really need to know:", "Seriously,", "I'm confused about"]
    for q in base_questions[:10]:
        variation = f"{random.choice(emphatic)} {q.lower()}"
        variations.append(variation)

    return variations

def generate_dataset(num_samples=1000):
    """Generate completely unique cursing dataset"""
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
    """Generate and save the cursing dataset"""
    print("Generating cursing dataset...")

    # Generate dataset
    dataset = generate_dataset(num_samples=1000)

    # Save
    output_path = "data/raw/cursing_dataset.jsonl"
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
