# {{ title }}

**Generated:** {{ generated_at }}
{% set report_data = report | default({}) %}
{% set readiness = report_data.readiness | default({}) %}
{% set source_summary = report_data.source_summary | default({}) %}
{% set evidence_items = report_data.included_items | default(items) %}
{% set excluded_items = report_data.excluded_items | default([]) %}

## What changed

{% for item in evidence_items %}
### {{ loop.index }}. {{ item.title or 'Untitled' }}

{% if item.summary %}{{ item.summary }}{% elif item.llm_summary %}{{ item.llm_summary }}{% endif %}

{% if item.published_at %}Published: {{ item.published_at }}{% endif %}
{% if item.url %}[Follow up]({{ item.url }}){% endif %}

{% endfor %}

## Timeline and recency

- Report readiness: {{ readiness.state | default('unknown') }}{% if readiness.score is defined %} ({{ readiness.score }}/100){% endif %}
- Included updates: {{ report_data.included_count | default(evidence_items | length) }}
- Generated at: {{ generated_at }}

{% if readiness.warnings | default([]) %}
## Evidence caveats

{% for warning in readiness.warnings %}
- {{ warning.message | default(warning.code | default('Evidence warning')) }}
{% endfor %}
{% endif %}

## Source diversity

- Unique sources: {{ source_summary.unique_source_count | default(0) }}
- Missing provenance: {{ source_summary.missing_source_count | default(0) }}

## Follow-up links

{% for item in evidence_items %}
{% if item.url %}- [{{ item.title or 'Untitled' }}]({{ item.url }}){% endif %}
{% endfor %}

## Excluded trail

{% if excluded_items %}
{% for item in excluded_items %}
- {{ item.title or ('Update #' ~ item.id) }} - {{ item.reason | default('excluded') }}
{% endfor %}
{% else %}
No excluded updates captured.
{% endif %}
