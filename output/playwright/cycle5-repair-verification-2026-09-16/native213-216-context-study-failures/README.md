# Native PostgreSQL context and Study Pack failures

Observed on API56113 source47e23bd5f3. StudyPack worker starts06:44:24; originaljob2 quarantined06:45:55 fromPostgreSQL persistencefailure216. OwnedAPI06:48:47 incorrectlyqueued/errornull215. ActualCharacter context logs expose213settingsQueryResult and214WorldBook unsupportedconnectioncontext. Repaired213/214commit04f1acb28c and215commit29ad7f2ba0 are notyetloaded;216 diagnosisactive. Redactederror excerpts include otherknownstartupwarnings, notallnewfindings. Sourcehash receiptsbound runtimebefore/afterhealth; dependenciesreused. No nativejob requeue/mutation.
Known runtime credentials and JWT/PEM patterns scanned, zero matches. Original evidence remains unchanged.
