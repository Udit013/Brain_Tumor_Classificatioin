import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { Assistant } from "./Assistant";

export const metadata: Metadata = { title: "Research assistant" };

export default function AssistantPage() {
  return (
    <>
      <PageHeader eyebrow="Research assistant" title="Ask the literature">
        Answers are assembled from 241 openly licensed PubMed Central articles and the project&apos;s own documents.
        Every statement cites a passage, and a deterministic check removes any statement its sources do not support.
        When the deployment enables the Gemini API free tier, your question and the retrieved public passages are sent
        to Google to draft the answer, and Google may use free-tier content to improve its products: don&apos;t include
        personal information. Without it, answers are quoted passages and nothing leaves the server.
      </PageHeader>
      <Assistant />
    </>
  );
}
