# infra/environment/dev

DVC remote (S3) と GitHub Actions 用 OIDC ロールを払い出す dev 環境の Terraform root module。

> **このリポジトリは public。** `terraform.tfstate` / `terraform.tfvars` は
> `.gitignore` 済みだが、commit 前に必ず staged 内容を確認すること。
>
> `terraform apply` / `destroy` は `.claude/settings.json` で **deny** されている。
> インフラ変更は人間の判断で実行する。Claude が実行してよいのは `fmt` / `validate` / `plan` のみ。

## 初回セットアップ

1. 管理権限を持つ IAM を AWS CLI プロファイル (例: `kaggriculture-admin`) として `~/.aws/credentials` に用意する。
2. 本ディレクトリに `terraform.tfvars` を作成 (`terraform.tfvars.example` をコピーして値を差し替え)。
   - `dvc_bucket_name` はグローバル一意。推奨: `kaggriculture-dvc-<AWS_ACCOUNT_ID>`
3. `AWS_PROFILE=kaggriculture-admin terraform init`
4. `AWS_PROFILE=kaggriculture-admin terraform plan`
5. 内容を確認した上で `AWS_PROFILE=kaggriculture-admin terraform apply`

## apply 後: DVC の設定

```bash
# 払い出された access key をローカルプロファイルに登録
AWS_PROFILE=kaggriculture aws configure set aws_access_key_id "$(terraform output -raw dvc_iam_access_key_id)"
AWS_PROFILE=kaggriculture aws configure set aws_secret_access_key "$(terraform output -raw dvc_iam_secret_access_key)"

# リポジトリルートから remote URL を実バケットに差し替える
# (初期値は s3://kaggriculture-dvc-CHANGEME/remote のプレースホルダ)
cd ../../..
./dev/dvc remote modify s3 url "s3://$(terraform -chdir=infra/environment/dev output -raw dvc_bucket_name)/remote"
./dev/dvc setup   # cache dir + profile を .dvc/config.local に書き込む (gitignored)
./dev/dvc push
```

## GitHub Actions OIDC

`module.github_actions_oidc` が払い出す IAM ロールを Actions が OIDC で assume する。
`scrape-kaggle.yml` はこのロールで DVC remote (`remote/*`) と実行ログ (`scrape_logs/*`)
を読み書きする。長期 AWS クレデンシャルは一切使わない (public リポジトリなので必須)。

apply 後、リポジトリ secret を登録する:

```bash
gh secret set AWS_ROLE_ARN --body "$(terraform output -raw github_actions_role_arn)"
gh secret set KAGGLE_USERNAME --body "<your-kaggle-username>"
gh secret set KAGGLE_KEY --body "<your-kaggle-key>"
```

## State 管理

初回は **local state** で bootstrap する (`versions.tf` に backend ブロック無し)。
運用に乗ったら state 専用 bucket を別途作成し、`backend "s3"` を追加して
`terraform init -migrate-state` で移行する。

**local state のファイルは絶対に commit しない。**

## 削除

```bash
AWS_PROFILE=kaggriculture-admin terraform destroy
```

**注意**: bucket 内にオブジェクトが残っていると destroy に失敗する。
事前に `aws s3 rm s3://<bucket> --recursive` が必要。
