import { expect, it } from 'vitest'
import { createIngestJobsTracker, pollTrackedIngestJobs } from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/tldw/ingest-jobs-orchestrator'
it.each(['unknown','cancelled'])('nested terminal %s must not be a clean completed result', async status => {
  const tracker=createIngestJobsTracker<{id:string}>();
  tracker.trackSubmit({batch_id:'probe',jobs:[{id:1}]},{id:'source'});
  const result=await pollTrackedIngestJobs({tracker,fetchJob:async()=>({ok:true,data:{status:'completed',result:{status,media_id:1,warnings:['Processing not confirmed']}}}),timeoutMs:1000,pollIntervalMs:1,isCancelled:()=>false,onCancel:async()=>{},mapCompleted:(_item,data)=>({kind:'completed',data}),mapFailure:(_item,details)=>({kind:'failed',data:details.data}),mapCancelled:()=>({kind:'cancelled',data:undefined})});
  expect(result[0].kind).not.toBe('completed');
});
