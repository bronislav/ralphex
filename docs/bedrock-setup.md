# AWS Bedrock Support

ralphex can run in Docker with Claude models hosted on AWS Bedrock instead of the Anthropic API. This is useful for organizations that require data residency, custom model access, or want to use their existing AWS infrastructure.

## Quick setup

1. Set environment variables:

```bash
export RALPHEX_USE_BEDROCK=1
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_PROFILE=ralphex-bedrock
export AWS_REGION=us-east-1
```

2. Login if using SSO:

```bash
aws sso login --profile=ralphex-bedrock
```

3. Run ralphex:

```bash
ralphex docs/plans/feature.md
```

## How it works

When `RALPHEX_USE_BEDROCK=1` is set:

- Keychain credential extraction is skipped (not needed for Bedrock)
- The `~/.claude` directory check is skipped
- AWS credentials are exported from the profile via `aws configure export-credentials`
- Bedrock-related environment variables are passed to the container

The wrapper never mounts `~/.aws` - credentials are exported as environment variables only.

## Environment variables

### Required

| Variable | Description |
|----------|-------------|
| `RALPHEX_USE_BEDROCK` | Set to `1` to enable Bedrock mode |
| `CLAUDE_CODE_USE_BEDROCK` | Set to `1` for Claude Code to use Bedrock |
| `AWS_REGION` | AWS region for Bedrock API (e.g., `us-east-1`) |

### Authentication (one of these)

| Variable | Description |
|----------|-------------|
| `AWS_PROFILE` | AWS profile name - credentials are exported via `aws configure export-credentials` |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | Explicit credentials (skip profile export) |

### Optional

| Variable | Description |
|----------|-------------|
| `AWS_SESSION_TOKEN` | Session token for temporary credentials |
| `ANTHROPIC_MODEL` | Override default model |
| `ANTHROPIC_SMALL_FAST_MODEL` | Model for quick operations |
| `DISABLE_PROMPT_CACHING` | Set to disable prompt caching |
| `RALPHEX_EXTRA_ENV` | Comma-separated list of additional env vars to pass |

## Security best practices

### Use a dedicated AWS profile

Create a separate AWS profile with minimal Bedrock-only permissions. This follows the principle of least privilege and limits exposure if credentials are compromised.

### Minimal IAM policy

Foundation models and inference profiles:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockInvokeFoundationModels",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
            ]
        },
        {
            "Sid": "BedrockInvokeInferenceProfiles",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*:*:inference-profile/*anthropic.claude*"
            ]
        }
    ]
}
```

More restrictive (specific region and models):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockInvokeClaudeFoundationModels",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-20250514-v1:0",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-20250514-v1:0"
            ]
        },
        {
            "Sid": "BedrockInvokeClaudeInferenceProfiles",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1:*:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
                "arn:aws:bedrock:us-east-1:*:inference-profile/us.anthropic.claude-haiku-4-20250514-v1:0"
            ]
        }
    ]
}
```

Inference profiles use cross-region prefixes (e.g., `us.anthropic.claude-*` for US regions). Check your Bedrock console for exact inference profile IDs available in your account.

## Setup options

### SSO with dedicated permission set

1. Create a permission set in AWS IAM Identity Center with the policy above
2. Assign to your user/group for the target account
3. Configure the profile:

```bash
aws configure sso --profile ralphex-bedrock
```

4. Login before running ralphex:

```bash
aws sso login --profile=ralphex-bedrock
```

### IAM user with dedicated policy

1. Create an IAM user with the policy above attached
2. Generate access keys for the user
3. Configure the profile:

```bash
aws configure --profile ralphex-bedrock
# enter access key ID and secret
```

## Example usage

### With SSO profile (recommended)

```bash
export RALPHEX_USE_BEDROCK=1
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_PROFILE=ralphex-bedrock
export AWS_REGION=us-east-1

# login if needed
aws sso login --profile=ralphex-bedrock

ralphex docs/plans/feature.md
```

### With explicit credentials

```bash
export RALPHEX_USE_BEDROCK=1
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...

ralphex docs/plans/feature.md
```

### With extra environment variables

```bash
export RALPHEX_USE_BEDROCK=1
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_PROFILE=ralphex-bedrock
export AWS_REGION=us-east-1

# pass additional env vars to container
export RALPHEX_EXTRA_ENV="CLAUDE_CODE_MAX_OUTPUT_TOKENS,MAX_THINKING_TOKENS"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000
export MAX_THINKING_TOKENS=10000

ralphex docs/plans/feature.md
```

## Startup output

With profile-based auth:

```
using image: ghcr.io/umputun/ralphex-go:latest
bedrock mode: enabled (keychain skipped)
  exporting credentials from profile: ralphex-bedrock
  passing: AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, CLAUDE_CODE_USE_BEDROCK
```

With explicit credentials:

```
using image: ghcr.io/umputun/ralphex-go:latest
bedrock mode: enabled (keychain skipped)
  using explicit credentials
  passing: AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, CLAUDE_CODE_USE_BEDROCK
```

## Troubleshooting

### "Unable to locate credentials"

The AWS CLI couldn't find valid credentials. Check:

1. If using a profile, verify it's configured: `aws configure list --profile ralphex-bedrock`
2. If using SSO, login first: `aws sso login --profile=ralphex-bedrock`
3. If using explicit creds, verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set

### "ExpiredToken" or "The security token included in the request is expired"

Your credentials have expired. Solutions:

1. SSO: run `aws sso login --profile=<profile>` again
2. IAM user with session token: refresh the token
3. IAM user with long-term credentials: verify the access key is still active

### "AccessDeniedException" on InvokeModel

The IAM policy doesn't allow access to the requested model. Check:

1. The policy includes the model you're trying to use
2. The region in the policy matches `AWS_REGION`
3. For inference profiles, verify the ARN format matches your account

### "Model not found" or "ValidationException"

The model ID is incorrect or not available in your region. Check:

1. The model is enabled in your Bedrock console
2. The region supports the requested model
3. For inference profiles, use the correct cross-region prefix (e.g., `us.` for US regions)

### Warning about missing CLAUDE_CODE_USE_BEDROCK

The wrapper detected `RALPHEX_USE_BEDROCK=1` but `CLAUDE_CODE_USE_BEDROCK` is not set. Claude Code inside the container needs this variable to use Bedrock. Set both:

```bash
export RALPHEX_USE_BEDROCK=1
export CLAUDE_CODE_USE_BEDROCK=1
```

### Warning about missing AWS_REGION

Bedrock requires a region to be specified. Set `AWS_REGION`:

```bash
export AWS_REGION=us-east-1
```

### Warning about missing credentials

Neither `AWS_PROFILE` nor `AWS_ACCESS_KEY_ID` is set. Provide credentials via one of:

```bash
# option 1: profile
export AWS_PROFILE=ralphex-bedrock

# option 2: explicit credentials
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
```

## Notes

- The `~/.aws` directory is never mounted - only exported credentials are passed as environment variables
- Profile-based credential export uses `aws configure export-credentials --profile <profile> --format env`
- If credential export fails, a warning is logged but execution continues (the container may still work with other credential sources)
- Bedrock mode skips macOS keychain extraction since Anthropic API credentials aren't needed
