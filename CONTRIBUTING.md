# Contributing to Gemma 4 vLLM Deployment

Thank you for your interest in helping resolve the Gemma 4 MoE deployment challenges! This repository documents real-world problems encountered during production deployment, and community contributions are essential to solving them.

## How to Contribute

### Reporting New Failure Modes

If you've encountered a failure mode not documented in the [Forensic Runbook](docs/FORENSIC_RUNBOOK.md):

1. Open an issue with the title: `[Failure Mode] Brief description`
2. Include:
   - **Error message** (exact, unedited)
   - **Stack trace** (full traceback)
   - **Environment** (vLLM version, transformers version, GPU type, container image)
   - **Root cause** (if identified)
   - **Workaround** (if found)

### Submitting Fixes

1. Fork this repository
2. Create a feature branch: `git checkout -b fix/description`
3. Make your changes
4. Test thoroughly
5. Submit a pull request with:
   - Clear description of the fix
   - Which failure mode(s) it addresses (reference the Forensic Runbook section)
   - Test results (container build logs, inference output)

### Priority Areas

We're actively seeking help in these areas (ordered by impact):

1. **LoRA for MoE** — Implementing `get_expert_mapping()` for `Gemma4ForConditionalGeneration` in vLLM
2. **NF4 Quantization** — Making bitsandbytes work with MoE expert layers
3. **Dependency Resolution** — Finding a compatible version matrix for vLLM + transformers + huggingface_hub
4. **Vision Chat Template** — Extending `chat_template.jinja` for multimodal inputs without Jinja2 `namespace()`
5. **Thinking Support** — Enabling Gemma 4's thinking/reasoning mode in the serving configuration

### Code Style

- Python: Follow PEP 8
- Shell: Use `shellcheck`-clean bash
- Documentation: Use clear, technical English with concrete error messages and version numbers
- Comments: Explain *why*, not just *what*

### Testing

- All container changes must include a successful `docker build` log
- All deployment script changes must include mock deployment output
- All documentation changes must be factually accurate with version numbers

## Communication

- **Issues** — For bug reports, feature requests, and failure mode documentation
- **Pull Requests** — For code and documentation fixes
- **Discussions** — For architecture proposals and brainstorming

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Acknowledgments

All contributors will be credited in the README.md acknowledgments section.

---

*This project was created and is maintained by [Daniel Manzela](https://github.com/Manzela). Original research conducted April 2026.*
