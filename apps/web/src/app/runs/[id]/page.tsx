import { RunViewer } from "@/components/run-viewer";

export default async function RunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <RunViewer runId={id} />;
}
