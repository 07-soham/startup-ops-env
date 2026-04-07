import React from 'react';
import { EmailInput as EmailInputType } from '../lib/api';
import { Plus, Trash2, Mail } from 'lucide-react';

interface EmailInputProps {
  emails: EmailInputType[];
  onEmailsChange: (emails: EmailInputType[]) => void;
}

export const EmailInput: React.FC<EmailInputProps> = ({ emails, onEmailsChange }) => {
  const addEmail = () => {
    onEmailsChange([
      ...emails,
      {
        text: '',
        sender: 'user@example.com',
        subject: 'New Email',
        thread_id: `thread_${emails.length + 1}`,
      },
    ]);
  };

  const removeEmail = (index: number) => {
    const newEmails = emails.filter((_, i) => i !== index);
    onEmailsChange(newEmails);
  };

  const updateEmail = (index: number, field: keyof EmailInputType, value: string) => {
    const newEmails = [...emails];
    newEmails[index] = { ...newEmails[index], [field]: value };
    onEmailsChange(newEmails);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="label flex items-center gap-2">
          <Mail size={16} />
          Emails
        </label>
        <button
          onClick={addEmail}
          className="btn-secondary flex items-center gap-1 text-sm"
        >
          <Plus size={16} />
          Add Email
        </button>
      </div>

      {emails.length === 0 && (
        <div className="text-gray-500 text-sm italic">No emails added yet. Click "Add Email" to create one.</div>
      )}

      {emails.map((email, index) => (
        <div key={index} className="card border border-gray-200 p-4">
          <div className="flex justify-between items-start mb-3">
            <span className="text-sm font-medium text-gray-500">Email #{index + 1}</span>
            <button
              onClick={() => removeEmail(index)}
              className="text-red-500 hover:text-red-700"
            >
              <Trash2 size={16} />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <label className="text-xs text-gray-500">Sender</label>
              <input
                type="text"
                value={email.sender}
                onChange={(e) => updateEmail(index, 'sender', e.target.value)}
                className="input text-sm"
                placeholder="sender@email.com"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">Thread ID</label>
              <input
                type="text"
                value={email.thread_id}
                onChange={(e) => updateEmail(index, 'thread_id', e.target.value)}
                className="input text-sm"
                placeholder="thread_1"
              />
            </div>
          </div>

          <div className="mb-3">
            <label className="text-xs text-gray-500">Subject</label>
            <input
              type="text"
              value={email.subject}
              onChange={(e) => updateEmail(index, 'subject', e.target.value)}
              className="input text-sm"
              placeholder="Email subject"
            />
          </div>

          <div>
            <label className="text-xs text-gray-500">Content</label>
            <textarea
              value={email.text}
              onChange={(e) => updateEmail(index, 'text', e.target.value)}
              className="input text-sm"
              rows={3}
              placeholder="Email body text..."
            />
          </div>
        </div>
      ))}
    </div>
  );
};
