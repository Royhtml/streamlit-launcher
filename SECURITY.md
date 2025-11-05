

# Security Policy

## Supported Versions

The following versions of `streamlit-launcher` are currently supported with security updates. Versions not listed as supported will not receive fixes, even for critical vulnerabilities.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | ✅                  |
| 5.0.x   | ❌ (End of Support) |
| 4.0.x   | ✅                  |
| < 4.0   | ❌                  |

Support generally follows a rolling release policy: only active major versions and the latest minor release branch receive security patches.

---

## Reporting a Vulnerability

We take the security of `streamlit-launcher` seriously. If you discover a bug or security vulnerability, we kindly request that you report it responsibly.

### How to Report

To report a vulnerability, please contact us through one of the following secure channels:

* **Email:** `security@streamlit-launcher.dev` *(placeholder, ubah sesuai email kamu)*
* **GitHub Security Advisory:** Use the **"Report a vulnerability"** feature on GitHub (preferred)

Please include the following information:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fixes (if any)
5. Your contact information for follow-up

Avoid publicly disclosing the issue before coordinating with us. Publicly posting vulnerabilities before a fix is issued may put users at risk.

### Responsible Disclosure Process

Once a report is submitted:

| Step                | Timeline                   | Description                                                                             |
| ------------------- | -------------------------- | --------------------------------------------------------------------------------------- |
| Acknowledge report  | within **72 hours**        | We confirm receipt and begin initial assessment                                         |
| Initial assessment  | within **5 business days** | Determine severity and plan remediation                                                 |
| Security fix issued | depends on severity        | Critical fixes may be released immediately; other patches follow standard release cycle |
| Public advisory     | after patch release        | A security advisory will be published with credit to the reporter (if desired)          |

For critical vulnerabilities that affect many users, we may coordinate a private patch release before publishing public details.

---

### Vulnerability Acceptance / Rejection

We may decline reports that:

* Are out of project scope (e.g., issues in third-party dependencies)
* Rely on unrealistic or contrived attack vectors
* Represent expected behavior (not bugs)
* Duplicate reports already submitted by others

In the case of a declined report, we will provide a justification.

### Scope of Responsibility

`streamlit-launcher` is a CLI and helper tool intended to simplify booting Streamlit applications. This project **does not take responsibility** for vulnerabilities in:

* Applications launched using this tool
* Streamlit framework or its internal security model
* Deployed infrastructure or hosting environments
* Third-party packages used by end-users

Users are responsible for securing their Streamlit app and hosting environment.

---

### Thanks and Acknowledgment

We deeply appreciate the security community and responsible researchers who help keep `streamlit-launcher` safe for everyone.
If you wish to be credited after a fix is released, please let us know.

---
