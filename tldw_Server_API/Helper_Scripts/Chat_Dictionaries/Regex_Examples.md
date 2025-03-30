# Pattern Matching
/(order|track)\s+(\d+)/: |
  Tracking order {{match.group(2)}}...
  Status: {{get_order_status(match.group(2))}}
---@@@---


/price\s+of\s+(\w+)/: |
  Current price of {{match.group(1)}}:
  ${{get_price(match.group(1))}}
---@@@---

