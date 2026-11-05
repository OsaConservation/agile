-- Run once in BigQuery Studio, processing location us-south1, before deployment.
-- Existing tables cannot add a REQUIRED column. New API writes require a name;
-- historical rows remain NULL rather than inventing reviewer identities.
ALTER TABLE `osaanurans.osafrogdataset.annotations`
ADD COLUMN IF NOT EXISTS annotator_name STRING;

SELECT column_name, data_type, is_nullable
FROM `osaanurans.osafrogdataset.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'annotations' AND column_name = 'annotator_name';
