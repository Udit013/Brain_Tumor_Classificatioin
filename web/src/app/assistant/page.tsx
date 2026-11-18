import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { Assistant } from "./Assistant";

export const metadata: Metadata = { title: "Research assistant" };

export default function AssistantPage() {
  return (
    <>
      <PageHeader eyebrow="Research assistant" title="Ask the literature">
        Answers come from 241 openly licensed PubMed Central articles and the project&apos;s own documents. Every
        statement cites a passage; a deterministic check removes anything its sources don&apos;t support.
      </PageHeader>
      <Assistant />
      <p className="mt-6 max-w-3xl text-xs leading-relaxed text-muted">
        Privacy: when the language model is enabled, your question and the retrieved public passages are sent to the
        Gemini API free tier, whose terms let Google use that content to improve its products. Don&apos;t include
        personal information. Without it, answers are quoted passages and nothing leaves the server.
      </p>
    </>
  );
}
