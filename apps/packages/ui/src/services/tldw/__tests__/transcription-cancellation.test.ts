import { afterEach, expect, it, vi } from "vitest";
import { tldwRequest } from "../request-core";
import { modelsAudioMethods } from "../domains/models-audio";

it.each([modelsAudioMethods.transcribeAudio])("B05 actual transcription adapter forwards its cancellation signal", async method => {
  const signal=new AbortController().signal;
  const upload=vi.fn().mockResolvedValue({text:"actual adapter result"});
  const context={ ensureConfigForRequest: vi.fn(), upload };
  const blob=new Blob(["unit audio"], {type:"audio/wav"});
  const requestScope={config:{serverUrl:"http://unit-server",authMode:"multi-user" as const},userId:42};
  await method.call(context, blob, {model:"parakeet", signal, requestScope});
  expect(upload.mock.calls[0][0]).toMatchObject({path:"/api/v1/audio/transcriptions", abortSignal:signal, timeoutMs:180000, servicePromptConfig:{serverUrl:"http://unit-server",expectedUserId:42}, headers:{"X-TLDW-Expected-User-ID":"42"}, fields:{model:"parakeet"}});
  expect(upload.mock.calls[0][0].fields).not.toHaveProperty("signal");
});

it.each([modelsAudioMethods.transcribeAudio])("B05 cancelled file read never reaches upload", async method => {
  const controller=new AbortController();
  const upload=vi.fn();
  const context={ ensureConfigForRequest: vi.fn(), upload };
  const blob=new Blob(["unit audio"]);
  vi.spyOn(blob,"arrayBuffer").mockImplementation(async () => {controller.abort(); return new ArrayBuffer(0);});
  await expect(method.call(context, blob, {signal:controller.signal})).rejects.toMatchObject({name:"AbortError"});
  expect(upload).not.toHaveBeenCalled();
});

afterEach(() => vi.useRealTimers());

it.each([modelsAudioMethods.transcribeAudio])("B05 timeout lets a valid 15-second transcription complete through actual request core", async method => {
  vi.useFakeTimers();
  let ready!: () => void;
  const started=new Promise<void>(resolve=>{ready=resolve;});
  let abortedAt: number | null=null;
  const begin=Date.now();
  const fetchFn=vi.fn((_url: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((resolve,reject)=>{
    const done=setTimeout(()=>resolve(new Response(JSON.stringify({text:"Completed real request-core unit response"}),{status:200,headers:{"content-type":"application/json"}})),15000);
    init?.signal?.addEventListener("abort",()=>{abortedAt=Date.now()-begin;clearTimeout(done);reject(new DOMException("Unit transport aborted","AbortError"));});
    ready();
  })) as typeof fetch;
  const context={ensureConfigForRequest:vi.fn(),upload:(init:{timeoutMs?:number;abortSignal?:AbortSignal})=>tldwRequest({
    path:"https://api.example.com/api/v1/audio/transcriptions",method:"POST",noAuth:true,timeoutMs:init.timeoutMs,abortSignal:init.abortSignal,body:new FormData()
  },{getConfig:async()=>({serverUrl:"https://api.example.com",requestTimeoutMs:10000}),fetchFn})};
  const pending=method.call(context,new Blob(["unit audio"]),{model:"parakeet"});
  await started;
  await vi.advanceTimersByTimeAsync(15000);
  const response=await pending;
  expect({status:response.status,abortedAt}).toEqual({status:200,abortedAt:null});
  expect(fetchFn).toHaveBeenCalledTimes(1);
});

it.each([modelsAudioMethods.transcribeAudio])("B05 timeout preserves an explicit transcription request override", async method => {
  const upload=vi.fn().mockResolvedValue({text:"unit override response"});
  await method.call({ensureConfigForRequest:vi.fn(),upload} as unknown as TldwApiClientBase,new Blob(["unit audio"]),{model:"parakeet",timeoutMs:45000});
  expect(upload.mock.calls[0][0].timeoutMs).toBe(45000);
});
