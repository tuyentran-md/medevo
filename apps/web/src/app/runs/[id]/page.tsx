import fs from "node:fs";
import path from "node:path";

import { RunViewer } from "@/components/run-viewer";

const staticReplayMode = process.env.NEXT_PUBLIC_MEDEVO_STATIC_REPLAY === "1";

export async function generateStaticParams() {
  if (!staticReplayMode) {
    return [];
  }
  const indexPath = path.join(process.cwd(), "public", "replays", "index.json");
  if (!fs.existsSync(indexPath)) {
    return [];
  }
  const items = JSON.parse(fs.readFileSync(indexPath, "utf-8")) as Array<{ run_id: string }>;
  return items.map((item) => ({ id: item.run_id }));
}

export default async function RunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <RunViewer runId={id} />;
}
