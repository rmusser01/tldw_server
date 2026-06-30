import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import PackList from '@web/components/vn-assets/PackList';

describe('PackList', () => {
  it('exposes selected pack state to assistive technology', () => {
    render(
      <PackList
        packs={[
          { id: 1, title: 'Primary Pack', primary_character_id: 7 },
          { id: 2, title: 'Backup Pack', primary_character_id: 8 },
        ]}
        selectedPackId={1}
        onSelectPack={() => undefined}
      />
    );

    expect(screen.getByRole('button', { name: /Primary Pack/ })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByRole('button', { name: /Backup Pack/ })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
  });
});
