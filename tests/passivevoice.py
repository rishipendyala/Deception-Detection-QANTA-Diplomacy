
# Run in terminal : python -m spacy download en_core_web_sm

# Load spaCy English model
import spacy
import language_tool_python

# Load models
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

def is_passive_advanced(sentence, verbose=False):
    doc = nlp(sentence)
    has_passive_aux = False
    has_nsubjpass = False
    has_participle_passive = False
    patterns_matched = []

    for token in doc:
        # Direct passive aux (e.g., was, is being, will be)
        if token.dep_ == "auxpass":
            has_passive_aux = True
            patterns_matched.append("auxpass")

        # Passive nominal subject
        if token.dep_ == "nsubjpass":
            has_nsubjpass = True
            patterns_matched.append("nsubjpass")

        # Verb in past participle form (VBN) with no agent
        if token.tag_ == "VBN" and token.dep_ in {"acl", "advcl", "relcl", "ROOT"}:
            # Check if it's part of a passive clause
            for child in token.children:
                if child.dep_ == "aux" and child.text.lower() in {"is", "was", "were", "been", "being"}:
                    has_participle_passive = True
                    patterns_matched.append("VBN + aux")

    if verbose:
        print(f"[DEBUG] Matched patterns: {patterns_matched}")

    # Final decision: return if any known passive pattern is detected
    return any([has_passive_aux, has_nsubjpass, has_participle_passive])


# Example usage
sentences = [
    # Clearly passive
    "The letter was written by the secretary late last night.",
    "The project was completed ahead of schedule by the engineering team.",
    "Mistakes were made during the negotiation phase.",
    "A decision was reached after several hours of discussion.",
    "The treaty was signed by both parties at the summit.",
    "The ambassador was recalled by the foreign ministry.",
    "The plan had been abandoned due to logistical issues.",
    "The case was handled with great care by the legal team.",
    "The funds were redirected without prior notice.",
    "The report was being edited when the deadline passed.",

    # Clearly active
    "The diplomat wrote a detailed letter last night.",
    "Engineers completed the project ahead of schedule.",
    "They made several mistakes during the negotiation phase.",
    "We reached a decision after several hours of discussion.",
    "Both parties signed the treaty at the summit.",
    "The foreign ministry recalled the ambassador.",
    "They abandoned the plan due to logistical issues.",
    "The legal team handled the case with great care.",
    "They redirected the funds without prior notice.",
    "He was editing the report when the deadline passed.",

    # Nuanced / Edge cases
    "The cake, which had been baked hours earlier, was eaten by the children.",
    "Though controversial, the policy was widely implemented.",
    "She was considered a strong candidate by the committee.",
    "The issue, being discussed for months, finally saw resolution.",
    "It is believed that he acted alone.",
    "Rumors were spread throughout the office, causing unrest.",
    "The emails have been sent, but no replies were received.",
    "A statement was issued, though it lacked key information.",
    "They were being watched, though they didn't know it.",
    "The room was filled with laughter and music.",
    "His actions were misinterpreted, leading to conflict.",
    "The laws were enforced strictly in that region.",
    "Everyone was invited, but few attended.",
    "The building, damaged in the storm, was rebuilt by the government.",
    "The results were published in an international journal.",
    "Nothing was said about the missing documents.",
    "The news had been delivered before the press release.",
    "The system was being updated during the outage.",
    "New rules were introduced without explanation.",
    "The meeting was postponed until further notice.",
]

for s in sentences:
    result = is_passive_advanced(s)
    print(f"[{'PASSIVE' if result else 'ACTIVE'}] {s}")
