# {{ title }}

**Generated:** {{ generated_at }}
**Job:** {{ job_name }} | **Run:** {{ run_id }}
{% set report_data = report | default({}) %}
{% set readiness = report_data.readiness | default({}) %}
{% set source_summary = report_data.source_summary | default({}) %}
{% set evidence_items = report_data.included_items | default(items) %}
{% set excluded_items = report_data.excluded_items | default([]) %}

## Executive summary

Report readiness: {{ readiness.state | default('unknown') }}{% if readiness.score is defined %} ({{ readiness.score }}/100){% endif %}

- Included evidence: {{ report_data.included_count | default(evidence_items | length) }}
- Alert matches: {{ report_data.alert_count | default(0) }}
- Unique sources: {{ source_summary.unique_source_count | default(0) }}
- Missing provenance: {{ source_summary.missing_source_count | default(0) }}

{% if readiness.warnings | default([]) %}
## Evidence caveats

{% for warning in readiness.warnings %}
- {{ warning.message | default(warning.code | default('Evidence warning')) }}
{% endfor %}
{% endif %}

## Key findings

{% for item in evidence_items %}
### {{ loop.index }}. {{ item.title or 'Untitled' }}

{% if item.summary %}{{ item.summary }}{% elif item.llm_summary %}{{ item.llm_summary }}{% endif %}

- Source: {{ item.source_name | default('Unknown source') }}
{% if item.published_at %}- Published: {{ item.published_at }}{% endif %}
{% if item.url %}- Link: {{ item.url }}{% endif %}

{% if item.alerts | default([]) %}
Alert evidence:
{% for alert in item.alerts %}
- {{ alert.rule_name | default('Alert') }} ({{ alert.severity | default('unknown') }}){% if alert.matched_text %}: {{ alert.matched_text }}{% endif %}
{% endfor %}
{% else %}
Alert evidence: none captured.
{% endif %}

{% endfor %}

## Evidence table

| Update | Source | Published | Alerts | Link |
| --- | --- | --- | --- | --- |
{% for item in evidence_items %}
| {{ item.title or 'Untitled' }} | {{ item.source_name | default('Unknown source') }} | {{ item.published_at | default('-') }} | {{ item.alerts | default([]) | length }} | {{ item.url | default('-') }} |
{% endfor %}

## Excluded trail

{% if excluded_items %}
{% for item in excluded_items %}
- {{ item.title or ('Update #' ~ item.id) }} - {{ item.reason | default('excluded') }}
{% endfor %}
{% else %}
No excluded updates captured.
{% endif %}
