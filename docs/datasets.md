# MedEvo dataset map

This note locks the current Group-A data slice and the next-step HRT dataset
target so the pipeline does not drift by ad hoc schema guesses.

## Current open Group-A slice

`services/worker/app/microdata.py` uses NHANES 2005-2006 because it is public,
stable, and directly usable inside the sandboxed code-exec loop.

Files currently used:

- `DEMO_D.xpt`: age and sex (`RIAGENDR`, `RIDAGEYR`)
- `RHQ_D.xpt`: female hormone exposure (`RHQ540`) and indication flags
  (`RHQ551A`, `RHQ551D`, `RHQ551E`)
- `MCQ_D.xpt`: prevalent cardiovascular disease history (`MCQ160C`,
  `MCQ160D`, `MCQ160E`, `MCQ160F`)
- `BPQ_D.xpt`: hypertension and cholesterol treatment history (`BPQ050A`,
  `BPQ080`, `BPQ100D`)
- `BPX_D.xpt`: measured blood pressure (`BPXSY1-4`, `BPXDI1-4`)
- `BMX_D.xpt`: body mass index (`BMXBMI`)
- `SMQ_D.xpt`: cigarette exposure (`SMQ020`)
- `HDL_D.xpt`: direct HDL cholesterol (`LBDHDD`)

Current endpoint:

- Women age `45+`
- Exposure = ever use of female hormones (`RHQ540`)
- Outcome = composite cardiometabolic burden:
  prevalent CVD OR measured hypertension/treatment OR low HDL OR treated high
  cholesterol
- Adjustment = direct standardization over age band, smoking, and obesity

This is intentionally a proxy slice. It is good enough to exercise the Group-A
agent/sandbox/provenance path, but it is not a replacement for WHI when the
target is the historical HRT guideline reversal.

## Target HRT dataset

Best-fit next dataset is WHI-CTOS through BioLINCC. Public metadata already
shows the forms MedEvo should map once access is granted:

- `Form 43 - Hormone Use`
- `Form 44 - Current Medications`
- `Form 54 - Change of Medications`
- `Form 121 - Report of Cardiovascular Outcome`
- `Form 126 - Report of Venous Thromboembolic Disease (HRT)`
- `Form 150 - Hormone Use Update WHI Extension`
- `Form 153 - Medication and Supplement Inventory`

Operational handoff for the future WHI adapter:

1. Resolve treatment arm / hormone exposure from the hormone-use and medication
   forms.
2. Join adjudicated cardiovascular and VTE outcomes.
3. Emit dataset-slice source IDs with participant-era provenance, never free
   text citations.
4. Keep the same Group-A sandbox contract: unsupported claims degrade to
   `UNGROUNDED`, never crash the run.

Official source pages:

- [NHANES 2005-2006 RHQ documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/RHQ_D.htm)
- [NHANES 2005-2006 MCQ documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/MCQ_D.htm)
- [NHANES 2005-2006 BPQ documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPQ_D.htm)
- [NHANES 2005-2006 BPX documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPX_D.htm)
- [NHANES 2005-2006 BMX documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BMX_D.htm)
- [NHANES 2005-2006 SMQ documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/SMQ_D.htm)
- [NHANES 2005-2006 HDL documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HDL_D.htm)
- [BioLINCC WHI-CTOS study page](https://biolincc.nhlbi.nih.gov/studies/whi_ctos/)
- [WHI clinical-trial forms index](https://biolincc.nhlbi.nih.gov/media/studies/whict/hidden/doc/whi/forms/whiforms.html)
- [WHI CTOS data guide](https://biolincc.nhlbi.nih.gov/media/studies/whi_ctos/WHI_CTOS_Data_Guide.pdf)
