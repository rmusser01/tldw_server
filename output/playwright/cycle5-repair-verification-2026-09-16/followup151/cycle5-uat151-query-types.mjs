import ts from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/node_modules/typescript/lib/typescript.js';
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2';
const configFile=ts.readConfigFile(root+'/apps/tldw-frontend/tsconfig.json',ts.sys.readFile);
const parsed=ts.parseJsonConfigFileContent(configFile.config,ts.sys,root+'/apps/tldw-frontend');
const file=root+'/apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts';
const program=ts.createProgram([file],{...parsed.options,noEmit:true,incremental:false});
const checker=program.getTypeChecker();
const source=program.getSourceFile(file);
const visit=node=>{if(ts.isFunctionDeclaration(node)&&node.name?.text==='useDecksQuery') {const signature=checker.getSignatureFromDeclaration(node);const type=checker.getReturnTypeOfSignature(signature);console.log('RETURN:',checker.typeToString(type,undefined,ts.TypeFormatFlags.NoTruncation));const prop=type.getProperty('data');console.log('DATA:',checker.typeToString(checker.getTypeOfSymbolAtLocation(prop,node),undefined,ts.TypeFormatFlags.NoTruncation));}ts.forEachChild(node,visit)};visit(source);
