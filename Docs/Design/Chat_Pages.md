# Chat Pages

## Table of Contents
- [Introduction](#introduction)
- [Chat Page Design](#chat-page-design)
- [LLM Chat Page](#llm-chat-page)
- [Character Chat Page](#character-chat-page)
- [Chat Dictionaries](#chat-dictionaries)



# Link Dump:
https://github.com/caspianmoon/memoripy
https://arxiv.org/abs/2407.03974
https://arxiv.org/abs/2408.13718
https://github.com/p-e-w/sorcery
https://artefact2.github.io/llm-sampling/
https://kalami.medium.com/prompt-decorators-a-simple-way-to-improve-ai-responses-c3f3c2579a8c
https://github.com/t41372/Open-LLM-VTuber
- Other Chat Front-ends:
    https://prompt.16x.engineer/
	https://www.librechat.ai/docs/features/fork
	https://github.com/danny-avila/LibreChat

### Introduction


### Chat Page Design



### LLM Chat Page



### Character Chat Page





### Chat Dictionaries
- **101**
  - In tldw, there is a functionality called 'Chat Dictionaries'.
  - These are essentially a list of words or phrases that are looked for and replaced with a matching word or phrase or sentence/paragraph/etc.
  - The idea is to allow for the LLM to gain context around certain words or phrases that may not be common knowledge.
  - An example might be replacing '101' with '101(is a term used to describe the basics of a subject)' (Although honestly I'd be surprised if 101 wasn't in the LLM's vocabulary.)
  - So with this, the LLM can be fed a sentence like "I need help with 101" and it can respond with "I can help you with the basics of a subject."
  - This is a very basic example, but the idea is to allow the LLM to understand and respond to a wider range of topics and questions.
- **Usage**
  - Chat dictionaries may be used from the config file or from the chat page itself.
  - They can be used to replace words or phrases with other words or phrases.
    * This can be used to help add context or replace acronyms with their full meaning.
  - Simply create a markdown file with the words/phrases you want to replace and what you want to replace them with.
  - Then either:
    - A: Move the file to the chat dictionary folder and modify the config file for the specific chat dictionary you want to replace
    - B: Upload the chat dictionary file from the chat page itself for use in that chat.
- **Implementation**
  - Chat dictionary files are markdown files with a specific structure.
  - The structure is extremely simple and easy to understand.
    - It is a key-value pair where the key is the word or phrase you want to replace and the value is what you want to replace it with
      - For single-line entries: separated by a colon `:`
      - For multi-line entries: separated by a colon `:` and then a pipe `|`
      - Each entry is separated by `---@@@---`
      - Example of a single line `greeting` entry which matches on `greeting` and replaces it with `Hello!`, while `weather` is replaced with a multi-line entry:
        ```
        greeting: Hello!
        weather: |
          Current: Sunny
          Temp: 75°F
        ---@@@---
        ```

- **Examples**
    - For examples, see the [Chat Dictionary User Manual Specification](#chat-dictionary-user-manual-specification) below.

# <a name="chat-dictionary-user-manual-specification"></a> Chat Dictionary User Manual Specification

## 1. Introduction
The **Chat Dictionary** dynamically modifies chat messages before sending them to an LLM. Use it for:
- Dynamic responses based on keywords/patterns
- Managing lore/world-building in role-playing
- Enforcing consistent terminology
- Adding context-aware behavior

---

## 2. Key Features
1. Text Replacement
2. Regex Support
3. Probability-Based Triggers (0-100%)
4. Group Conflict Resolution
5. Token Budget Management
6. Timed Effects (cooldowns/delays)
7. Replacement Limits

---

## 3. File Format
Save entries in `.md` files with this structure:

### 3.1 Basic Syntax
```
Single-line:
    key: value
```

```
Multi-line:
    key: |
      line1
      line2
---@@@---
```

### 3.2 Advanced Syntax
Regex pattern:
    /pattern/: replacement

Add properties:
    key: value | probability=50 group=global

---

## 4. Entry Properties
| Property         | Description                          | Default |
|------------------|--------------------------------------|---------|
| key              | Word/phrase/regex to match           | Required|
| content          | Replacement text                     | Required|
| probability      | Trigger chance (0-100)               | 100     |
| group            | Category for conflict resolution     | None    |
| timed_effects    | cooldown/delay/sticky (seconds)      | None    |
| max_replacements | Maximum uses                         | 1       |

---

## 5. Usage Examples

### Simple Replacement
    greeting: Hello!

### Regex Replacement
    /\b\d{3}-\d{4}\b/: REDACTED-PHONE

### Multi-Line Entry
```
    weather: |
      Current: Sunny
      Temp: 75°F
---@@@---
```
### Timed Entry
    alert: System overload! | timed_effects={"cooldown":60}

---

## 6. Integration Steps
1. Upload `.md` files via UI
2. Configure settings:
   - Max Tokens (500-2000)
   - Strategy (sorted_evenly/character_first/global_first)
3. Replacements auto-apply to user input

---

## 7. Troubleshooting

**Problem**            | **Solution**
-----------------------|-----------------------------
No replacement         | Check key spelling/regex
Token limit exceeded   | Reduce content length
Unexpected behavior    | Check group conflicts
File not loading       | Validate markdown syntax

---

## 8. Full Example File

`Chat_Dictionary.md`
```
# Character Info
hero: John | group=character probability=90

# Global Rules
/\[redacted\]/: [CLASSIFIED] | group=global

# Multi-line
backstory: |
  John grew up in a small village.
  He became a hero after defeating the dragon.
---@@@---

# Timed Effect
warning: System critical! | timed_effects={"cooldown":120}
```



