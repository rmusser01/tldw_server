import React from "react";
import { Modal, Input, Tabs, Empty, Tooltip } from "antd";
import { useTranslation } from "react-i18next";
import { useStoreChatModelSettings } from "@/store/model";
import {
  BookText,
  Code2,
  FileJson,
  MessageSquare,
  Pen,
  Search,
  Shield,
  Sparkles,
  Target,
  Users,
  Wand2,
  GraduationCap,
  Briefcase,
  Bot,
  Copy,
  Check,
} from "lucide-react";

export type PromptCategory =
  | "general"
  | "coding"
  | "writing"
  | "analysis"
  | "roleplay"
  | "specialized";

export type PromptTemplate = {
  id: string;
  title: string;
  description: string;
  category: PromptCategory;
  content: string;
  icon: React.ReactNode;
  tags: string[];
};

export type PromptTemplateMetadata = Omit<PromptTemplate, "icon">;

const PROMPT_TEMPLATES: PromptTemplate[] = [
  // General
  {
    id: "helpful-assistant",
    title: "Helpful Assistant",
    description: "A friendly and knowledgeable general-purpose assistant",
    category: "general",
    content: `You are a helpful, harmless, and honest AI assistant. Your goal is to provide accurate, useful, and well-reasoned responses to user queries.

Guidelines:
- Be concise but thorough
- Acknowledge uncertainty when appropriate
- Provide sources or reasoning when making claims
- Ask clarifying questions if the request is ambiguous
- Be respectful and professional`,
    icon: <Bot className="h-4 w-4" />,
    tags: ["default", "general", "helpful"],
  },
  {
    id: "concise-responder",
    title: "Concise Responder",
    description: "Get straight-to-the-point answers without fluff",
    category: "general",
    content: `You are a concise assistant. Provide direct, to-the-point answers without unnecessary elaboration.

Rules:
- No preamble or filler phrases
- Use bullet points for multiple items
- Skip pleasantries unless specifically asked
- If a simple yes/no suffices, provide just that with brief reasoning`,
    icon: <Target className="h-4 w-4" />,
    tags: ["concise", "direct", "minimal"],
  },
  {
    id: "socratic-teacher",
    title: "Socratic Teacher",
    description: "Learn through guided questions rather than direct answers",
    category: "general",
    content: `You are a Socratic teacher. Instead of providing direct answers, guide the user to discover solutions themselves through thoughtful questions.

Approach:
- Ask probing questions that lead to understanding
- Break down complex problems into smaller parts
- Encourage critical thinking
- Provide hints when the user is stuck, but let them connect the dots
- Celebrate when they reach insights on their own`,
    icon: <GraduationCap className="h-4 w-4" />,
    tags: ["teaching", "learning", "socratic"],
  },

  // Coding
  {
    id: "code-expert",
    title: "Code Expert",
    description: "Expert programmer with best practices focus",
    category: "coding",
    content: `You are an expert software developer with deep knowledge across multiple programming languages and paradigms.

Guidelines:
- Write clean, well-documented code
- Follow industry best practices and design patterns
- Consider edge cases and error handling
- Explain your reasoning and trade-offs
- Suggest improvements and optimizations
- Use appropriate data structures and algorithms`,
    icon: <Code2 className="h-4 w-4" />,
    tags: ["programming", "development", "code"],
  },
  {
    id: "code-reviewer",
    title: "Code Reviewer",
    description: "Review code for bugs, style, and improvements",
    category: "coding",
    content: `You are a senior code reviewer. Analyze code for:

1. **Bugs & Issues**: Logic errors, race conditions, security vulnerabilities
2. **Code Quality**: Readability, maintainability, DRY principles
3. **Performance**: Time/space complexity, unnecessary operations
4. **Best Practices**: Language idioms, design patterns, testing
5. **Documentation**: Comments, naming, clarity

Provide specific, actionable feedback with examples of improved code.`,
    icon: <Search className="h-4 w-4" />,
    tags: ["review", "quality", "bugs"],
  },
  {
    id: "debugging-assistant",
    title: "Debugging Assistant",
    description: "Help identify and fix bugs systematically",
    category: "coding",
    content: `You are a debugging expert. Help identify and fix bugs using systematic approaches:

Process:
1. Understand the expected vs actual behavior
2. Identify potential root causes
3. Suggest debugging strategies (logging, breakpoints, etc.)
4. Propose fixes with explanations
5. Recommend tests to prevent regression

Ask clarifying questions about error messages, stack traces, and reproduction steps.`,
    icon: <Target className="h-4 w-4" />,
    tags: ["debugging", "bugs", "troubleshooting"],
  },

  // Writing
  {
    id: "creative-writer",
    title: "Creative Writer",
    description: "Craft engaging stories and creative content",
    category: "writing",
    content: `You are a creative writing assistant with expertise in fiction, poetry, and storytelling.

Capabilities:
- Develop compelling characters and dialogue
- Create vivid descriptions and settings
- Maintain consistent tone and voice
- Structure narratives with proper pacing
- Offer constructive feedback on writing
- Adapt to various genres and styles`,
    icon: <Sparkles className="h-4 w-4" />,
    tags: ["creative", "fiction", "storytelling"],
  },
  {
    id: "technical-writer",
    title: "Technical Writer",
    description: "Create clear documentation and technical content",
    category: "writing",
    content: `You are a technical writing expert. Create clear, accurate, and well-structured documentation.

Standards:
- Use plain language and avoid jargon when possible
- Include examples and code snippets where relevant
- Structure content with clear headings and sections
- Define technical terms on first use
- Use consistent formatting and terminology
- Consider the target audience's expertise level`,
    icon: <BookText className="h-4 w-4" />,
    tags: ["documentation", "technical", "clarity"],
  },
  {
    id: "editor-proofreader",
    title: "Editor & Proofreader",
    description: "Polish and improve existing text",
    category: "writing",
    content: `You are a professional editor and proofreader. Review and improve text for:

- Grammar, spelling, and punctuation
- Clarity and conciseness
- Tone and voice consistency
- Flow and structure
- Word choice and style

Provide tracked changes with explanations, or offer suggestions while preserving the author's voice.`,
    icon: <Pen className="h-4 w-4" />,
    tags: ["editing", "proofreading", "polish"],
  },

  // Analysis
  {
    id: "data-analyst",
    title: "Data Analyst",
    description: "Analyze data and extract insights",
    category: "analysis",
    content: `You are a data analysis expert. Help interpret data and extract meaningful insights.

Approach:
- Identify patterns and trends
- Calculate relevant statistics
- Create clear visualizations (describe them)
- Draw actionable conclusions
- Note limitations and caveats
- Suggest follow-up analyses`,
    icon: <Target className="h-4 w-4" />,
    tags: ["data", "statistics", "insights"],
  },
  {
    id: "research-assistant",
    title: "Research Assistant",
    description: "Help with research and literature review",
    category: "analysis",
    content: `You are a research assistant helping with academic and professional research.

Capabilities:
- Summarize complex papers and articles
- Identify key findings and methodology
- Compare different sources and viewpoints
- Suggest related research directions
- Help formulate research questions
- Note limitations and biases in sources`,
    icon: <Search className="h-4 w-4" />,
    tags: ["research", "academic", "literature"],
  },

  // Roleplay
  {
    id: "character-actor",
    title: "Character Actor",
    description: "Roleplay as a specific character or persona",
    category: "roleplay",
    content: `You are a skilled character actor. When given a character description or persona:

- Stay in character consistently
- Respond as the character would
- Maintain their speech patterns and mannerisms
- Use their knowledge and perspective
- React authentically to situations

Signal clearly when breaking character if needed for clarification.`,
    icon: <Users className="h-4 w-4" />,
    tags: ["roleplay", "character", "acting"],
  },
  {
    id: "interview-prep",
    title: "Interview Prep Coach",
    description: "Practice interviews with feedback",
    category: "roleplay",
    content: `You are an interview preparation coach. Help users practice for job interviews.

Modes:
1. **Mock Interviewer**: Ask realistic interview questions and provide feedback
2. **Answer Coach**: Help craft compelling answers using STAR method
3. **Technical Prep**: Practice technical questions with hints and explanations

Provide constructive feedback on communication, content, and confidence.`,
    icon: <Briefcase className="h-4 w-4" />,
    tags: ["interview", "career", "practice"],
  },

  // Specialized
  {
    id: "json-generator",
    title: "JSON Generator",
    description: "Output structured data in JSON format",
    category: "specialized",
    content: `You are a structured data generator. Always respond with valid JSON.

Requirements:
- Output only valid JSON (no markdown code blocks unless explicitly requested)
- Include all relevant fields
- Use consistent naming conventions (camelCase)
- Handle arrays and nested objects appropriately
- Validate data types

If the request is unclear, ask for the expected schema first.`,
    icon: <FileJson className="h-4 w-4" />,
    tags: ["json", "structured", "data"],
  },
  {
    id: "security-reviewer",
    title: "Security Reviewer",
    description: "Analyze code and systems for security issues",
    category: "specialized",
    content: `You are a cybersecurity expert. Review code and systems for security vulnerabilities.

Focus areas:
- OWASP Top 10 vulnerabilities
- Authentication and authorization flaws
- Input validation and sanitization
- Sensitive data exposure
- Security misconfigurations
- Dependency vulnerabilities

Provide severity ratings and remediation guidance.`,
    icon: <Shield className="h-4 w-4" />,
    tags: ["security", "vulnerabilities", "audit"],
  },
  {
    id: "prompt-engineer",
    title: "Prompt Engineer",
    description: "Help craft effective prompts for AI systems",
    category: "specialized",
    content: `You are a prompt engineering expert. Help users create effective prompts for AI systems.

Guidance:
- Clarify the desired output format
- Add relevant context and constraints
- Use clear, unambiguous language
- Include examples when helpful
- Structure complex prompts logically
- Iterate and refine based on results`,
    icon: <Wand2 className="h-4 w-4" />,
    tags: ["prompts", "ai", "engineering"],
  },
];

const CATEGORY_INFO: Record<
  PromptCategory,
  { label: string; icon: React.ReactNode }
> = {
  general: { label: "General", icon: <MessageSquare className="h-4 w-4" /> },
  coding: { label: "Coding", icon: <Code2 className="h-4 w-4" /> },
  writing: { label: "Writing", icon: <Pen className="h-4 w-4" /> },
  analysis: { label: "Analysis", icon: <Search className="h-4 w-4" /> },
  roleplay: { label: "Roleplay", icon: <Users className="h-4 w-4" /> },
  specialized: { label: "Specialized", icon: <Wand2 className="h-4 w-4" /> },
};

const toPromptTemplateMetadata = ({
  icon: _icon,
  ...template
}: PromptTemplate): PromptTemplateMetadata => template;

export const getPromptTemplateById = (
  id: string | null | undefined,
): PromptTemplateMetadata | null => {
  if (!id) return null;
  const template = PROMPT_TEMPLATES.find((entry) => entry.id === id);
  return template ? toPromptTemplateMetadata(template) : null;
};

type Props = {
  open: boolean;
  onClose: () => void;
  onSelect: (template: PromptTemplate) => void;
};

export const SystemPromptTemplatesModal: React.FC<Props> = ({
  open,
  onClose,
  onSelect,
}) => {
  const { t } = useTranslation(["playground", "common"]);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [activeCategory, setActiveCategory] = React.useState<string>("all");
  const [copiedId, setCopiedId] = React.useState<string | null>(null);

  const filteredTemplates = React.useMemo(() => {
    let templates = PROMPT_TEMPLATES;

    if (activeCategory !== "all") {
      templates = templates.filter((t) => t.category === activeCategory);
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      templates = templates.filter(
        (t) =>
          t.title.toLowerCase().includes(query) ||
          t.description.toLowerCase().includes(query) ||
          t.tags.some((tag) => tag.includes(query)),
      );
    }

    return templates;
  }, [activeCategory, searchQuery]);

  const handleCopy = async (template: PromptTemplate) => {
    try {
      await navigator.clipboard.writeText(template.content);
      setCopiedId(template.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error("Failed to copy prompt template", error);
      setCopiedId(null);
    }
  };

  const handleSelect = (template: PromptTemplate) => {
    onSelect(template);
    onClose();
  };

  const tabItems = [
    { key: "all", label: t("playground:templates.all", "All") },
    ...Object.entries(CATEGORY_INFO).map(([key, { label, icon }]) => ({
      key,
      label: (
        <div className="flex items-center gap-1.5">
          {icon}
          <span>{t(`playground:templates.category.${key}`, label)}</span>
        </div>
      ),
    })),
  ];

  return (
    <Modal
      title={
        <div className="flex items-center gap-2">
          <BookText className="h-5 w-5 text-primary" />
          {t("playground:templates.title", "System prompts")}
        </div>
      }
      open={open}
      onCancel={onClose}
      footer={null}
      width={700}
      className="prompt-templates-modal"
    >
      <div className="space-y-4">
        <Input
          placeholder={t(
            "playground:templates.search",
            "Search system prompts...",
          )}
          prefix={<Search className="h-4 w-4 text-text-subtle" />}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          allowClear
        />

        <Tabs
          activeKey={activeCategory}
          onChange={setActiveCategory}
          items={tabItems}
          size="small"
        />

        <div className="max-h-[400px] space-y-2 overflow-y-auto pr-1">
          {filteredTemplates.length === 0 ? (
            <Empty
              description={t(
                "playground:templates.noResults",
                "No system prompts found",
              )}
            />
          ) : (
            filteredTemplates.map((template) => (
              <div
                key={template.id}
                className="group rounded-lg border border-border bg-surface p-3 transition-colors hover:border-primary/50 hover:bg-surface-hover"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 rounded-lg bg-primary/10 p-2 text-primary">
                      {template.icon}
                    </div>
                    <div>
                      <div className="font-medium">{template.title}</div>
                      <div className="mt-0.5 text-sm text-text-muted">
                        {template.description}
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {template.tags.map((tag) => (
                          <span
                            key={tag}
                            className="rounded bg-surface-hover px-1.5 py-0.5 text-[10px] text-text-subtle"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <Tooltip
                      title={t("playground:templates.copy", "Copy prompt")}
                    >
                      <button
                        type="button"
                        onClick={() => handleCopy(template)}
                        className="rounded p-1.5 text-text-muted hover:bg-surface hover:text-text"
                      >
                        {copiedId === template.id ? (
                          <Check className="h-4 w-4 text-success" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </button>
                    </Tooltip>
                    <Tooltip
                      title={t("playground:templates.use", "Use prompt")}
                    >
                      <button
                        type="button"
                        onClick={() => handleSelect(template)}
                        className="rounded bg-primary px-2 py-1 text-xs font-medium text-surface hover:bg-primaryStrong"
                      >
                        {t("common:use", "Use")}
                      </button>
                    </Tooltip>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </Modal>
  );
};

type TemplatesButtonProps = {
  onSelect: (template: PromptTemplate) => void;
  className?: string;
};

export const SystemPromptTemplatesButton: React.FC<TemplatesButtonProps> = ({
  onSelect,
  className,
}) => {
  const { t } = useTranslation(["playground"]);
  const [open, setOpen] = React.useState(false);

  return (
    <>
      <Tooltip title={t("playground:templates.title", "System prompts")}>
        <button
          type="button"
          onClick={() => setOpen(true)}
          className={`flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-sm text-text-muted transition-colors hover:border-primary/50 hover:bg-surface-hover hover:text-text ${className || ""}`}
        >
          <BookText className="h-4 w-4" />
          <span className="hidden sm:inline">
            {t("playground:templates.button", "System prompts")}
          </span>
        </button>
      </Tooltip>
      <SystemPromptTemplatesModal
        open={open}
        onClose={() => setOpen(false)}
        onSelect={onSelect}
      />
    </>
  );
};

export { PROMPT_TEMPLATES, CATEGORY_INFO };
