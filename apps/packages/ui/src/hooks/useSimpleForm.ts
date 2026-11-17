import React from "react"

type FormErrors<T> = Partial<Record<keyof T, string>>
type Validator<T> = (value: T[keyof T], values: T) => string | null
type Validators<T> = Partial<Record<keyof T, Validator<T>>>

type UseSimpleFormOptions<T> = {
  initialValues: T
  validate?: Validators<T>
}

const shallowEqual = <T extends Record<string, any>>(a: T, b: T): boolean => {
  if (a === b) return true
  const aKeys = Object.keys(a)
  const bKeys = Object.keys(b)
  if (aKeys.length !== bKeys.length) return false
  for (const key of aKeys) {
    if (!Object.prototype.hasOwnProperty.call(b, key)) return false
    if (!Object.is(a[key], b[key])) return false
  }
  return true
}

export const useSimpleForm = <T extends Record<string, any>>({
  initialValues,
  validate
}: UseSimpleFormOptions<T>) => {
  const initialRef = React.useRef(initialValues)
  const [values, setValuesState] = React.useState<T>(initialValues)
  const liveValues = React.useRef(initialValues)
  const revision = React.useRef(0)
  const commitValues = React.useCallback((next: T) => {
    if (!shallowEqual(liveValues.current, next)) revision.current++
    liveValues.current = next
    setValuesState(next)
  }, [])
  const getValues = React.useCallback(() => ({ ...liveValues.current }), [])
  const getRevision = React.useCallback(() => revision.current, [])
  const [errors, setErrors] = React.useState<FormErrors<T>>({})
  const [dirtyBaseline, setDirtyBaseline] = React.useState<T>(initialValues)

  const setValues = React.useCallback(
    (next: Partial<T>) => {
      commitValues({ ...liveValues.current, ...next })
    },
    [commitValues]
  )

  const setFieldValue = React.useCallback(
    <K extends keyof T>(field: K, value: T[K]) => {
      commitValues({ ...liveValues.current, [field]: value })
    },
    [commitValues]
  )

  const setFieldError = React.useCallback(
    <K extends keyof T>(field: K, error: string) => {
      setErrors((prev) => ({ ...prev, [field]: error }))
    },
    []
  )

  const clearFieldError = React.useCallback(<K extends keyof T>(field: K) => {
    setErrors((prev) => {
      if (!prev[field]) return prev
      const next = { ...prev }
      delete next[field]
      return next
    })
  }, [])

  const reset = React.useCallback(() => {
    commitValues(initialRef.current)
    setErrors({})
    setDirtyBaseline(initialRef.current)
  }, [commitValues])

  const resetDirty = React.useCallback((nextValues?: T) => {
    setDirtyBaseline(nextValues ?? values)
  }, [values])

  const isDirty = React.useCallback(() => {
    return !shallowEqual(values, dirtyBaseline)
  }, [dirtyBaseline, values])

  const runValidation = React.useCallback(() => {
    if (!validate) return null
    const nextErrors: FormErrors<T> = {}
    let hasError = false
    for (const key of Object.keys(validate) as Array<keyof T>) {
      const validator = validate[key]
      if (!validator) continue
      const result = validator(values[key], values)
      if (result) {
        nextErrors[key] = result
        hasError = true
      }
    }
    setErrors(nextErrors)
    return hasError ? nextErrors : null
  }, [validate, values])

  const onSubmit = React.useCallback(
    (handler: (values: T) => void | Promise<void>) =>
      async (event?: React.FormEvent) => {
        if (event?.preventDefault) {
          event.preventDefault()
        }
        const validationErrors = runValidation()
        if (validationErrors) return
        await handler(values)
      },
    [runValidation, values]
  )

  const getInputProps = React.useCallback(
    <K extends keyof T>(
      field: K,
      options?: { type?: "checkbox" }
    ) => {
      const isCheckbox = options?.type === "checkbox"
      return {
        value: isCheckbox ? undefined : (values[field] as T[K]),
        checked: isCheckbox ? Boolean(values[field]) : undefined,
        onChange: (event: any) => {
          if (isCheckbox) {
            const next =
              typeof event === "boolean"
                ? event
                : Boolean(event?.target?.checked)
            setFieldValue(field, next as T[K])
            return
          }
          if (event && event.target) {
            setFieldValue(field, event.target.value as T[K])
            return
          }
          setFieldValue(field, event as T[K])
        }
      }
    },
    [setFieldValue, values]
  )

  return {
    values,
    getValues,
    getRevision,
    errors,
    setValues,
    setFieldValue,
    setFieldError,
    clearFieldError,
    reset,
    resetDirty,
    isDirty,
    onSubmit,
    getInputProps
  }
}
