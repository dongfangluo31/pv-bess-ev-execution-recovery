from protocol import digest
FIELDS = ('seq', 'issue_s', 'effective_s', 'valid_until_s', 'signed_kw', 'control_decision_s', 'dispatch_slot_s', 'ready_s')

def summarize_queue(queue_state):

    def command_summary(command):
        if command is None:
            return None
        fields = {key: command[key] for key in FIELDS if key in command}
        return dict(**fields, original_command_sha256=digest(command))
    return dict(as_of_s=queue_state['as_of_s'], last_delivered_command=command_summary(queue_state.get('last_delivered_command')), pending_commands=[command_summary(command) for command in queue_state['pending_commands']])
