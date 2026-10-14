/**
 * Shared display formatting for the admin tables (C-S5).
 *
 * One `Intl.DateTimeFormat` instance is constructed at module scope and
 * reused by every admin table cell: constructing a formatter (and its
 * locale/timezone resolution) per rendered cell showed up as significant
 * allocation churn on the larger admin tables. Cells call
 * {@link formatAdminDateTime} instead of `new Date(v).toLocaleString()`.
 */

/** Shared medium-date/short-time formatter for admin table cells. */
export const adminDateTime = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short"
})

/** Format an ISO string, epoch number, or Date through the shared formatter. */
export const formatAdminDateTime = (v: string | number | Date): string =>
  adminDateTime.format(new Date(v))
