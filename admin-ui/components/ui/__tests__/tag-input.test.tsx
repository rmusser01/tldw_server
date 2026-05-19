import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TagInput } from '../tag-input';

describe('TagInput', () => {
  let onChange: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onChange = vi.fn();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders empty with placeholder', () => {
    render(<TagInput value="" onChange={onChange} placeholder="Add tags" />);

    const input = screen.getByRole('textbox', { name: 'Add tags' });
    expect(input).toBeInTheDocument();
    expect(input.getAttribute('placeholder')).toBe('Add tags');
  });

  it('adds a tag when Enter is pressed', async () => {
    const user = userEvent.setup();
    render(<TagInput value="" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'hello{Enter}');

    expect(onChange).toHaveBeenCalledWith('hello');
  });

  it('adds a tag when comma key is pressed', async () => {
    const user = userEvent.setup();
    render(<TagInput value="" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'world,');

    expect(onChange).toHaveBeenCalledWith('world');
  });

  it('removes a tag when X button is clicked', async () => {
    const user = userEvent.setup();
    render(<TagInput value="alpha, beta, gamma" onChange={onChange} />);

    // All three tags should be visible
    expect(screen.getByText('alpha')).toBeInTheDocument();
    expect(screen.getByText('beta')).toBeInTheDocument();
    expect(screen.getByText('gamma')).toBeInTheDocument();

    // Click remove on "beta"
    await user.click(screen.getByLabelText('Remove beta'));

    expect(onChange).toHaveBeenCalledWith('alpha, gamma');
  });

  it('creates multiple tags when pasting comma-separated text', async () => {
    const user = userEvent.setup();
    render(<TagInput value="" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.click(input);
    await user.paste('foo, bar, baz');

    expect(onChange).toHaveBeenCalledWith('foo, bar, baz');
  });

  it('deduplicates entries - does not add existing tag', async () => {
    const user = userEvent.setup();
    render(<TagInput value="alpha" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'alpha{Enter}');

    // onChange should NOT have been called because "alpha" already exists
    expect(onChange).not.toHaveBeenCalled();
  });

  it('fires onChange with comma-separated string when tag is added', async () => {
    const user = userEvent.setup();
    render(<TagInput value="existing" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'newTag{Enter}');

    expect(onChange).toHaveBeenCalledWith('existing, newTag');
  });

  it('removes last tag via Backspace when input is empty', async () => {
    const user = userEvent.setup();
    render(<TagInput value="first, second" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.click(input);
    await user.keyboard('{Backspace}');

    expect(onChange).toHaveBeenCalledWith('first');
  });

  it('hides placeholder when tags exist', () => {
    render(<TagInput value="tag1" onChange={onChange} placeholder="Add tags" />);

    const input = screen.getByRole('textbox');
    expect(input.getAttribute('placeholder')).toBe('');
  });
});
