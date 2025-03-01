import json
import random
import argparse
from typing import List, Dict, Any

# Define categories for the fake dataset
CATEGORIES = [
    "code",
    "mathematics",
    "instruction-following",
    "structured-data",
    "general-knowledge",
    "reasoning"
]

# Sample prompts for each category
CODE_PROMPTS = [
    "Write a Python function to find the nth Fibonacci number using dynamic programming.",
    "Create a JavaScript function that sorts an array of objects by a specific property.",
    "Write a SQL query to find the top 5 customers who have spent the most money.",
    "Implement a binary search tree in C++ with insert, delete, and search operations.",
    "Create a simple REST API using Flask that handles CRUD operations for a user entity.",
    "Write a regular expression to validate email addresses.",
    "Implement a quicksort algorithm in Python.",
    "Create a React component that displays a paginated list of items.",
    "Write a bash script that finds and deletes files older than 30 days.",
    "Implement a simple neural network from scratch using NumPy."
]

MATH_PROMPTS = [
    "Solve the equation: 3x^2 + 5x - 2 = 0",
    "Calculate the derivative of f(x) = x^3 * ln(x)",
    "Find the area of a circle with radius 5 units.",
    "Solve the system of equations: 2x + y = 5 and 3x - 2y = 4",
    "Calculate the probability of drawing 2 aces from a standard deck of cards without replacement.",
    "Find the volume of a sphere with radius 3 units.",
    "Calculate the limit of (sin(x)/x) as x approaches 0.",
    "Find the eigenvalues of the matrix [[3, 1], [2, 2]].",
    "Solve the differential equation: dy/dx = 2xy with y(0) = 1",
    "Calculate the expected value of rolling two fair dice and taking their sum."
]

INSTRUCTION_PROMPTS = [
    "Explain how to change a flat tire on a car.",
    "Provide step-by-step instructions for making chocolate chip cookies.",
    "Explain how to set up a basic home network.",
    "Provide instructions for creating a strong password.",
    "Explain how to perform CPR on an adult.",
    "Provide steps to troubleshoot a computer that won't turn on.",
    "Explain how to plant and care for a vegetable garden.",
    "Provide instructions for creating a budget spreadsheet.",
    "Explain how to properly cite sources in APA format.",
    "Provide steps to prepare for a job interview."
]

STRUCTURED_DATA_PROMPTS = [
    "Convert this JSON to XML: {\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
    "Parse this CSV data and calculate the average of the second column: \"name,score,date\\nAlice,85,2023-01-01\\nBob,92,2023-01-02\\nCharlie,78,2023-01-03\"",
    "Extract all email addresses from this text: \"Contact us at support@example.com or sales@example.com for more information.\"",
    "Format this data as a markdown table: [{\"name\":\"Alice\",\"age\":25,\"role\":\"Developer\"},{\"name\":\"Bob\",\"age\":30,\"role\":\"Designer\"}]",
    "Convert this date string \"2023-04-15T14:30:00Z\" to the format \"April 15, 2023 at 2:30 PM UTC\".",
    "Parse this log entry and extract the timestamp, level, and message: \"2023-04-15 14:30:45 ERROR Failed to connect to database: Connection refused\"",
    "Convert this XML to JSON: \"<person><name>John</name><age>30</age><city>New York</city></person>\"",
    "Extract the domain name from this URL: \"https://www.example.com/products?id=123\"",
    "Parse this YAML and convert it to JSON: \"name: John\\nage: 30\\naddress:\\n  city: New York\\n  zip: 10001\"",
    "Format this phone number \"1234567890\" as \"(123) 456-7890\"."
]

KNOWLEDGE_PROMPTS = [
    "Explain the process of photosynthesis in plants.",
    "Describe the main causes of World War I.",
    "Explain how the human immune system works.",
    "Describe the key features of quantum mechanics.",
    "Explain the process of DNA replication.",
    "Describe the major events of the French Revolution.",
    "Explain how a blockchain works.",
    "Describe the structure and function of the human brain.",
    "Explain the theory of relativity in simple terms.",
    "Describe the water cycle and its importance to Earth's ecosystems."
]

REASONING_PROMPTS = [
    "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
    "A train travels from city A to city B at 60 mph and returns at 40 mph. What is the average speed for the entire journey?",
    "If you have a cake and cut it into 8 equal pieces, then eat 3 pieces, what fraction of the cake remains?",
    "A store is having a 20% off sale. If you have a coupon for an additional 15% off, what is the total percentage discount?",
    "If 5 people can paint 5 houses in 5 days, how many people would be needed to paint 10 houses in 10 days?",
    "A car travels uphill at 30 mph and downhill at 60 mph. What is the average speed for the round trip?",
    "If the probability of rain on Saturday is 40% and the probability of rain on Sunday is 30%, what is the probability that it will rain on either Saturday or Sunday?",
    "A clock shows 3:15. What is the angle between the hour and minute hands?"
]

ALL_PROMPTS = {
    "code": CODE_PROMPTS,
    "mathematics": MATH_PROMPTS,
    "instruction-following": INSTRUCTION_PROMPTS,
    "structured-data": STRUCTURED_DATA_PROMPTS,
    "general-knowledge": KNOWLEDGE_PROMPTS,
    "reasoning": REASONING_PROMPTS
}

def generate_response(prompt: str, category: str) -> str:
    """
    Generate a fake response for a given prompt and category.
    In a real scenario, these would be high-quality responses from a language model or human.
    """
    if category == "code":
        if "Python" in prompt and "Fibonacci" in prompt:
            return """
def fibonacci(n):
    '''Calculate the nth Fibonacci number using dynamic programming.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    # Initialize array to store Fibonacci numbers
    fib = [0] * (n + 1)
    fib[1] = 1
    
    # Fill the array using bottom-up approach
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    
    return fib[n]

# Example usage:
# print(fibonacci(10))  # Output: 55
"""
        elif "JavaScript" in prompt and "sorts" in prompt:
            return """
function sortArrayByProperty(array, property) {
  // Create a copy of the array to avoid modifying the original
  return [...array].sort((a, b) => {
    if (a[property] < b[property]) {
      return -1;
    }
    if (a[property] > b[property]) {
      return 1;
    }
    return 0;
  });
}

// Example usage:
// const people = [
//   { name: 'Alice', age: 30 },
//   { name: 'Bob', age: 25 },
//   { name: 'Charlie', age: 35 }
// ];
// const sortedByAge = sortArrayByProperty(people, 'age');
// console.log(sortedByAge);
"""
        else:
            return "Here's a solution to your coding problem:\n\n```\n// Implementation would go here\n// This is a placeholder for a detailed code solution\n```\n\nThe above code implements the requested functionality efficiently. Let me know if you need any clarification or have questions about the implementation."
    
    elif category == "mathematics":
        if "3x^2 + 5x - 2 = 0" in prompt:
            return """
To solve the quadratic equation 3x^2 + 5x - 2 = 0, I'll use the quadratic formula:

x = (-b ± √(b² - 4ac)) / 2a

Where a = 3, b = 5, and c = -2

x = (-5 ± √(5² - 4 × 3 × (-2))) / (2 × 3)
x = (-5 ± √(25 + 24)) / 6
x = (-5 ± √49) / 6
x = (-5 ± 7) / 6

This gives us two solutions:
x₁ = (-5 + 7) / 6 = 2/6 = 1/3
x₂ = (-5 - 7) / 6 = -12/6 = -2

Therefore, the solutions to the equation 3x^2 + 5x - 2 = 0 are x = 1/3 and x = -2.
"""
        else:
            return "To solve this mathematical problem, I'll work through it step by step:\n\n[Step-by-step solution would be provided here]\n\nTherefore, the answer is [final result]. This solution uses [relevant mathematical concept] and follows from [explanation of approach]."
    
    elif category == "instruction-following":
        if "flat tire" in prompt:
            return """
# How to Change a Flat Tire

## Materials Needed:
- Spare tire
- Jack
- Lug wrench
- Wheel wedges (or bricks/large rocks)

## Step-by-Step Instructions:

1. **Find a Safe Location**
   - Pull over to a flat, stable surface away from traffic
   - Turn on your hazard lights

2. **Prepare Your Vehicle**
   - Apply the parking brake
   - Place wheel wedges in front of and behind tires opposite to the flat tire
   - Remove the hubcap or wheel cover (if applicable)

3. **Loosen the Lug Nuts**
   - Use the lug wrench to loosen (but not remove) the lug nuts by turning counterclockwise
   - Only loosen them about ¼ to ½ turn at this point

4. **Position the Jack**
   - Find the proper jacking point near the flat tire (consult your owner's manual)
   - Place the jack under the vehicle at the appropriate spot

5. **Raise the Vehicle**
   - Use the jack to raise the vehicle until the flat tire is about 6 inches off the ground
   - Never put any part of your body under the vehicle while it's jacked up

6. **Remove the Lug Nuts and Tire**
   - Completely remove the lug nuts and place them in a secure location
   - Pull the flat tire straight toward you to remove it
   - Set it aside

7. **Mount the Spare Tire**
   - Line up the spare tire with the wheel bolts
   - Push it all the way onto the wheel mount

8. **Replace the Lug Nuts**
   - Put the lug nuts back on and tighten them by hand
   - Tighten in a star pattern (not circle) for even pressure

9. **Lower the Vehicle**
   - Carefully lower the vehicle using the jack until the spare tire touches the ground
   - Remove the jack completely

10. **Fully Tighten the Lug Nuts**
    - Use the lug wrench to fully tighten the lug nuts in a star pattern
    - Apply significant pressure to ensure they're secure

11. **Final Steps**
    - Replace the hubcap if it fits the spare
    - Store the flat tire, jack, and lug wrench in your vehicle
    - Check the spare tire pressure when possible

Remember to get your flat tire repaired or replaced as soon as possible, as most spare tires are not designed for long-term use or high speeds.
"""
        else:
            return "Here are the step-by-step instructions:\n\n1. First, [initial step]\n2. Next, [second step]\n3. Then, [third step]\n...\n\nImportant tips:\n- [Relevant tip 1]\n- [Relevant tip 2]\n\nFollow these instructions carefully to achieve the best results."
    
    elif category == "structured-data":
        if "JSON to XML" in prompt:
            return """
The JSON:
```json
{"name": "John", "age": 30, "city": "New York"}
```

Converted to XML:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<root>
  <name>John</name>
  <age>30</age>
  <city>New York</city>
</root>
```

The conversion process maps each JSON key to an XML element, with the corresponding value as the element's content. The root element is added to satisfy XML's requirement for a single root element.
"""
        else:
            return "I've processed the structured data as requested:\n\n```\n[Processed data result would go here]\n```\n\nThe processing involved [explanation of the transformation applied]. This format is useful for [explanation of benefits of the new format]."
    
    elif category == "general-knowledge":
        if "photosynthesis" in prompt:
            return """
# Photosynthesis in Plants

Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose (or other sugars).

## The Basic Equation

The overall chemical equation for photosynthesis is:

6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂

In words: Carbon dioxide + water + light energy → glucose + oxygen

## The Process in Detail

Photosynthesis occurs in two main stages:

1. **Light-Dependent Reactions**
   - Take place in the thylakoid membrane of the chloroplasts
   - Capture energy from sunlight
   - Convert it to chemical energy in the form of ATP and NADPH
   - Produce oxygen as a byproduct from the splitting of water molecules

2. **Light-Independent Reactions (Calvin Cycle)**
   - Occur in the stroma of the chloroplasts
   - Use the ATP and NADPH from the light-dependent reactions
   - Capture and reduce carbon dioxide to produce glucose
   - Recycle ADP and NADP+ to be used again in light-dependent reactions

## Key Components

- **Chlorophyll**: The primary pigment that absorbs light energy
- **Chloroplasts**: Specialized organelles where photosynthesis occurs
- **Stomata**: Pores in leaves that allow CO₂ to enter and O₂ to exit
- **Water**: Obtained through the plant's roots
- **Carbon Dioxide**: Absorbed from the atmosphere through stomata

## Importance of Photosynthesis

- Produces oxygen necessary for aerobic organisms
- Creates glucose that serves as the foundation of most food chains
- Removes carbon dioxide from the atmosphere
- Provides the energy that drives ecosystems

Photosynthesis is essential for life on Earth as it is the primary process that converts inorganic carbon into organic compounds that can be used by living organisms.
"""
        else:
            return "Here's an explanation of this topic:\n\nThe [topic] is [comprehensive explanation with key facts, historical context, and scientific principles]. The main aspects to understand are [key points]. This is significant because [explanation of importance or relevance]."
    
    elif category == "reasoning":
        if "bat and ball" in prompt:
            return """
Let's solve this step by step:

1) We know that a bat and ball cost $1.10 in total.
2) We also know that the bat costs $1.00 more than the ball.

Let's define variables:
- Let x = cost of the ball in dollars
- Then the bat costs (x + 1.00) dollars

From the first condition:
x + (x + 1.00) = 1.10
2x + 1.00 = 1.10
2x = 0.10
x = 0.05

Therefore:
- The ball costs $0.05
- The bat costs $1.05 (which is $1.00 more than the ball)

We can verify this is correct:
- Ball + Bat = $0.05 + $1.05 = $1.10 ✓
- Bat - Ball = $1.05 - $0.05 = $1.00 ✓

The ball costs $0.05.
"""
        else:
            return "To solve this reasoning problem, I'll analyze it logically:\n\n1. First, I'll identify what we know: [list of given information]\n2. Next, I'll determine what we need to find: [statement of the question]\n3. [Step-by-step reasoning process]\n\nTherefore, the answer is [conclusion]. This follows from [explanation of the logical steps]."
    
    else:
        return "Here is a detailed response to your question. [Detailed response would be provided here based on the specific prompt.]"

def generate_conversation(category: str) -> Dict[str, Any]:
    """Generate a single conversation with a prompt and response."""
    prompts = ALL_PROMPTS[category]
    prompt = random.choice(prompts)
    response = generate_response(prompt, category)
    
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ],
        "category": category
    }

def generate_multi_turn_conversation(category: str, num_turns: int = 3) -> Dict[str, Any]:
    """Generate a multi-turn conversation with multiple prompts and responses."""
    prompts = ALL_PROMPTS[category]
    prompt = random.choice(prompts)
    response = generate_response(prompt, category)
    
    conversations = [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": response}
    ]
    
    # Generate additional turns
    for _ in range(num_turns - 1):
        follow_up = f"Can you explain more about the {random.choice(['first', 'second', 'last', 'main', 'key'])} part?"
        follow_up_response = f"Certainly! Here's more detail about that part: [Additional explanation would be provided here, expanding on the previous response with more specific details, examples, or clarifications related to the requested aspect.]"
        
        conversations.extend([
            {"from": "human", "value": follow_up},
            {"from": "gpt", "value": follow_up_response}
        ])
    
    return {
        "conversations": conversations,
        "category": category
    }

def generate_dataset(num_examples: int, multi_turn_ratio: float = 0.3) -> List[Dict[str, Any]]:
    """Generate a dataset with a mix of single-turn and multi-turn conversations."""
    dataset = []
    
    # Calculate number of multi-turn conversations
    num_multi_turn = int(num_examples * multi_turn_ratio)
    num_single_turn = num_examples - num_multi_turn
    
    # Generate single-turn conversations
    for _ in range(num_single_turn):
        category = random.choice(CATEGORIES)
        conversation = generate_conversation(category)
        dataset.append(conversation)
    
    # Generate multi-turn conversations
    for _ in range(num_multi_turn):
        category = random.choice(CATEGORIES)
        num_turns = random.randint(2, 4)  # Random number of turns between 2 and 4
        conversation = generate_multi_turn_conversation(category, num_turns)
        dataset.append(conversation)
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Generate a fake dataset for LLaDA SFT")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="sft_data/conversations.json", help="Output file path")
    parser.add_argument("--multi_turn_ratio", type=float, default=0.3, help="Ratio of multi-turn conversations")
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate the dataset
    dataset = generate_dataset(args.num_examples, args.multi_turn_ratio)
    
    # Save the dataset to a JSON file
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {args.num_examples} examples and saved to {args.output}")
    print(f"- Single-turn conversations: {int(args.num_examples * (1 - args.multi_turn_ratio))}")
    print(f"- Multi-turn conversations: {int(args.num_examples * args.multi_turn_ratio)}")

if __name__ == "__main__":
    main()
