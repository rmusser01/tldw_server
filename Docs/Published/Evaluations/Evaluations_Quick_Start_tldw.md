# Evaluations Quick Start - 5 Minutes to Your First Evaluation

This guide gets you evaluating AI content in just 5 minutes using tldw_server's evaluation module.

## 🎯 What You'll Learn
- Evaluate an AI-generated summary
- Check Q&A response quality
- Interpret evaluation scores

## 📋 Prerequisites

✅ tldw_server is running (default: `http://localhost:8000`)
✅ You have content to evaluate (we'll provide examples)
✅ (Optional) OpenAI API key for advanced evaluations

## 🚀 3 Quick Examples

### Example 1: Evaluate a Summary (2 minutes)

**What we're doing**: Checking if an AI summary is good quality.

```python
import requests

# Your content
original_article = """
Climate change is one of the most pressing issues of our time.
Rising global temperatures are causing melting ice caps, rising sea levels,
and more frequent extreme weather events. Scientists agree that human
activities, particularly the burning of fossil fuels, are the primary cause.
"""

ai_summary = """
Climate change, driven by human activities like fossil fuel use,
is causing rising temperatures and extreme weather events.
"""

# Evaluate the summary
response = requests.post(
    "http://localhost:8000/api/v1/evaluations/geval",
    json={
        "source_text": original_article,
        "summary": ai_summary,
        "api_name": "openai"  # or "anthropic", "google", etc.
    }
)

result = response.json()
print(f"Summary Quality Score: {result['average_score']*100:.0f}%")
print(f"Assessment: {result['summary_assessment']}")
```

**Expected Output**:
```
Summary Quality Score: 85%
Assessment: High-quality summary with good factual consistency
```

### Example 2: Test Q&A Quality (2 minutes)

**What we're doing**: Checking if a chatbot gave a good answer.

```python
# Question and Answer scenario
question = "How do I reset my password?"

# What the chatbot found in documentation
context = ["To reset your password: 1) Click 'Forgot Password' on login page,
           2) Enter your email, 3) Check email for reset link,
           4) Create new password"]

# What the chatbot answered
bot_answer = "To reset your password, click the 'Forgot Password' link on
             the login page and follow the email instructions."

# Evaluate the response
response = requests.post(
    "http://localhost:8000/api/v1/evaluations/rag",
    json={
        "query": question,
        "retrieved_contexts": context,
        "generated_response": bot_answer,
        "api_name": "openai"
    }
)

result = response.json()
print(f"Answer Quality: {result['overall_score']*100:.0f}%")
print(f"Suggestions: {', '.join(result['suggestions'])}")
```

**Expected Output**:
```
Answer Quality: 92%
Suggestions: Consider mentioning all steps for completeness
```

### Example 3: Batch Evaluation (1 minute)

**What we're doing**: Evaluating multiple summaries at once.

```python
# Multiple items to evaluate
items = [
    {
        "source_text": "Article 1 full text...",
        "summary": "Article 1 summary..."
    },
    {
        "source_text": "Article 2 full text...",
        "summary": "Article 2 summary..."
    }
]

# Evaluate batch
response = requests.post(
    "http://localhost:8000/api/v1/evaluations/batch",
    json={
        "evaluation_type": "geval",
        "items": items,
        "api_name": "openai"
    }
)

result = response.json()
print(f"Evaluated {result['summary']['total_items']} items")
print(f"Average Score: {result['summary']['average_score']*100:.0f}%")
print(f"Success Rate: {result['summary']['successful']}/{result['summary']['total_items']}")
```

## 📊 Understanding Scores

| Score Range | Quality | What It Means |
|------------|---------|---------------|
| 90-100% | 🌟 Excellent | Production ready |
| 80-89% | ✅ Good | Minor improvements needed |
| 70-79% | ⚠️ Fair | Review recommended |
| Below 70% | ❌ Poor | Needs improvement |

## 🛠️ Quick Setup with cURL

If you prefer command line:

```bash
# Evaluate a summary
curl -X POST http://localhost:8000/api/v1/evaluations/geval \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "Your article here...",
    "summary": "Your summary here...",
    "api_name": "openai"
  }'

# Check evaluation health
curl http://localhost:8000/api/v1/health/evaluations
```

## 🎨 Try It With Your Content

Replace the example content with your own:

1. **For Summaries**: Use your article and its summary
2. **For Q&A**: Use your question and answer pairs
3. **For General Content**: Use any prompt and response

## 🚦 Next Steps

**Ready for more?**
- [Full User Guide](./User_Guides/Server/Evaluations_End_User_Guide.md) - Detailed explanations
- [API Reference](./API-related/tldw_Evaluations_API_Reference.md) - All endpoints
- [Best Practices](./User_Guides/Server/Evaluations_End_User_Guide.md#best-practices) - Pro tips

**Common Tasks**:
- Evaluate customer service responses
- Test different AI models
- Quality check generated content
- A/B test prompts

## ⚡ Quick Tips

1. **Start Simple**: Try one evaluation before batch processing
2. **Use Appropriate Models**: mini tiers for testing, flagship tiers for production
3. **Check Rate Limits**: 10 requests/minute for standard endpoints
4. **Monitor Costs**: Each evaluation costs ~$0.001-0.01 depending on model

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Rate limit exceeded" | Wait 60 seconds or use batch endpoint |
| "API key invalid" | Check your OpenAI/Anthropic key in config |
| "Low scores" | Check if source content is clear and complete |
| "Timeout error" | Reduce text size or increase timeout |

## 📝 Complete Working Example

Here's a full script you can copy and run:

```python
#!/usr/bin/env python3
"""
Quick evaluation example - ready to run!
Save as 'quick_eval.py' and run: python quick_eval.py
"""

import requests
import json

def evaluate_summary():
    """Evaluate a news summary"""

    # Real example content
    original = """
    Apple announced its latest iPhone 15 series today, featuring a new
    titanium design, improved camera system with 48MP main sensor, and
    the new A17 Pro chip. The phones start at $799 for the base model
    and go up to $1,199 for the Pro Max. The company also introduced
    USB-C charging, replacing the Lightning port after 11 years.
    """

    summary = """
    Apple unveiled the iPhone 15 with titanium build, 48MP camera,
    A17 Pro chip, and USB-C charging. Prices range from $799-$1,199.
    """

    # Send for evaluation
    response = requests.post(
        "http://localhost:8000/api/v1/evaluations/geval",
        json={
            "source_text": original,
            "summary": summary,
            "api_name": "openai"
        }
    )

    if response.status_code == 200:
        result = response.json()

        print("📊 EVALUATION RESULTS")
        print("="*40)
        print(f"Overall Score: {result['average_score']*100:.0f}%")
        print(f"\nDetailed Scores:")

        for metric_name, metric_data in result['metrics'].items():
            score = metric_data['score']
            stars = "⭐" * int(score * 5)
            print(f"  {metric_name.capitalize()}: {score:.2f} {stars}")

        print(f"\nAssessment: {result['summary_assessment']}")
        print(f"Time taken: {result['evaluation_time']:.2f} seconds")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("🚀 Running Quick Evaluation Example\n")
    evaluate_summary()
    print("\n✅ Done! Try it with your own content.")
```

## 🎉 Congratulations!

You've successfully run your first evaluation! You now know how to:
- ✅ Evaluate summaries with G-Eval
- ✅ Test Q&A responses with RAG evaluation
- ✅ Process multiple items with batch evaluation
- ✅ Interpret quality scores

**Time to evaluate your own AI content!** 🚀
