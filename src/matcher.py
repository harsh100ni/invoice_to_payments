import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import re  # Import regular expressions

# Optional fuzzy lib; use if you want
try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None  # type: ignore


@dataclass
class Match:
    payment_id: str
    invoice_id: str
    confidence: float
    rationale: str
    # Store matched amount for auditing
    amount_matched: float 


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df


def write_out(df: pd.DataFrame, path: str) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def baseline_normalize_name(name: Optional[str]) -> str:
    """A more robust name normalizer."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Handle common legal terms
    # e.g. "Acme Pvt. Ltd." -> "acme pvt ltd"
    name = re.sub(r'(\s|,)(pvt\.? ltd\.?|private limited)$', ' pvt ltd', name, flags=re.I)
    # e.g. "Acme Limited" -> "acme ltd"
    name = re.sub(r'(\s|,)(ltd\.?|limited)$', ' ltd', name, flags=re.I)
    # e.g. "Zeta Foods LLC" -> "zeta foods"
    name = re.sub(r'(\s|,)(llc\.?|inc\.?|co\.?)$', '', name, flags=re.I)
    # Handle specific abbreviations from data, e.g. "Gamma Ind."
    name = name.replace('ind.', 'industries')
    # Clean up remaining punctuation
    name = name.replace('.', '').replace(',', '')
    return name.strip()


def match_records(invoices: pd.DataFrame, payments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Implementation of a waterfall matching logic."""

    # --- 1. Preparation ---
    inv_df = invoices.copy()
    pay_df = payments.copy()

    # Convert types
    inv_df["amount"] = pd.to_numeric(inv_df["invoice_amount"], errors="coerce")
    pay_df["amount"] = pd.to_numeric(pay_df["payment_amount"], errors="coerce")
    
    inv_df["date"] = pd.to_datetime(inv_df["invoice_date"], errors="coerce", utc=True)
    pay_df["date"] = pd.to_datetime(pay_df["payment_date"], errors="coerce", utc=True)

    # Use the new, better normalizer
    inv_df['norm_name'] = inv_df['customer_name'].apply(baseline_normalize_name)
    pay_df['norm_name'] = pay_df['payer_name'].apply(baseline_normalize_name)

    inv_df["amount_remaining"] = inv_df["amount"]
    pay_df["amount_remaining"] = pay_df["amount"]
    
    matches_list = []
    TOLERANCE = 0.01

    # --- 2. Pass 1: Direct ID/Ref in memo (Confidence 1.0) ---
    inv_df = inv_df.reset_index(drop=True)
    pay_df = pay_df.reset_index(drop=True)

    for inv_idx, invoice in inv_df.iterrows():
        inv_rem = inv_df.at[inv_idx, 'amount_remaining']
        if inv_rem < TOLERANCE:
            continue

        # Build a set of all possible search terms from the invoice
        terms = set()
        if pd.notna(invoice['invoice_id']):
            terms.add(invoice['invoice_id'])
            # Add the number-only part, e.g., "1004" from "INV-1004"
            number_part = invoice['invoice_id'].split('-')[-1]
            if number_part.isdigit():
                 terms.add(number_part)
        if pd.notna(invoice['po_number']):
            terms.add(invoice['po_number'])
        if pd.notna(invoice['customer_ref']):
            terms.add(invoice['customer_ref'])
        
        terms = {str(t) for t in terms if pd.notna(t) and str(t)}

        if not terms:
            continue
            
        pattern = '|'.join(re.escape(t) for t in terms)

        potential_pay_indices = pay_df[
            pay_df['memo'].str.contains(pattern, na=False, case=False) &
            (pay_df['amount_remaining'] > TOLERANCE)
        ].index

        for pay_idx in potential_pay_indices:
            pay_rem = pay_df.at[pay_idx, 'amount_remaining']
            inv_rem = inv_df.at[inv_idx, 'amount_remaining'] # Re-check

            if pay_rem < TOLERANCE or inv_rem < TOLERANCE:
                continue

            match_amount = min(pay_rem, inv_rem)
            
            pay_df.at[pay_idx, 'amount_remaining'] -= match_amount
            inv_df.at[inv_idx, 'amount_remaining'] -= match_amount

            matches_list.append(Match(
                payment_id=pay_df.at[pay_idx, 'payment_id'],
                invoice_id=invoice['invoice_id'],
                confidence=1.0,
                rationale=f"Direct reference in memo (e.g., ID, PO, Ref)",
                amount_matched=round(match_amount, 2)
            ))

    # --- 3. Pass 2: Exact Amount, Name, Currency, Near Date (Confidence 0.9) ---
    unmatched_inv_idx = inv_df[inv_df['amount_remaining'] > TOLERANCE].index
    unmatched_pay_idx = pay_df[pay_df['amount_remaining'] > TOLERANCE].index

    if not unmatched_inv_idx.empty and not unmatched_pay_idx.empty:
        merged = pd.merge(
            pay_df.loc[unmatched_pay_idx],
            inv_df.loc[unmatched_inv_idx],
            on=['norm_name', 'amount', 'currency'],
            suffixes=('_pay', '_inv')
        )

        # *** FIX: Widen the date window to be more generous ***
        date_window = pd.Timedelta(days=31)
        if 'date_pay' in merged.columns:
            merged['date_diff'] = (merged['date_pay'] - merged['date_inv']).abs()
            good_matches = merged[merged['date_diff'] <= date_window].copy()
        else:
            good_matches = pd.DataFrame(columns=merged.columns)

        if not good_matches.empty:
            good_matches = good_matches.sort_values(by='date_diff')
            
            good_matches = good_matches.drop_duplicates(subset=['payment_id'], keep='first')
            good_matches = good_matches.drop_duplicates(subset=['invoice_id'], keep='first')

            for _, row in good_matches.iterrows():
                pay_idx = row['payment_id_pay_index']
                inv_idx = row['invoice_id_inv_index']
                
                pay_rem = pay_df.at[pay_idx, 'amount_remaining']
                inv_rem = inv_df.at[inv_idx, 'amount_remaining']

                match_amount = min(pay_rem, inv_rem)
                
                if match_amount < TOLERANCE:
                    continue

                pay_df.at[pay_idx, 'amount_remaining'] -= match_amount
                inv_df.at[inv_idx, 'amount_remaining'] -= match_amount

                matches_list.append(Match(
                    payment_id=row['payment_id'],
                    invoice_id=row['invoice_id'],
                    confidence=0.9,
                    rationale=f"Exact amount, name, currency, and date within {date_window.days} days",
                    amount_matched=round(match_amount, 2)
                ))

    # --- 4. Pass 3: Partial/Overpayment on Name/Currency (Confidence 0.75) ---
    unmatched_inv_idx = inv_df[inv_df['amount_remaining'] > TOLERANCE].index
    unmatched_pay_idx = pay_df[pay_df['amount_remaining'] > TOLERANCE].index
    
    if not unmatched_inv_idx.empty and not unmatched_pay_idx.empty:
        merged_partial = pd.merge(
            pay_df.loc[unmatched_pay_idx],
            inv_df.loc[unmatched_inv_idx],
            on=['norm_name', 'currency'],
            suffixes=('_pay', '_inv')
        )
        
        # *** FIX: Use a reasonable window for partials too ***
        date_window_partial = pd.Timedelta(days=31) 
        if 'date_pay' in merged_partial.columns:
            merged_partial['date_diff'] = (merged_partial['date_pay'] - merged_partial['date_inv']).abs()
            good_partials = merged_partial[merged_partial['date_diff'] <= date_window_partial].copy()
        else:
            good_partials = pd.DataFrame(columns=merged_partial.columns)

        if not good_partials.empty:
            good_partials = good_partials.sort_values(by=['payment_id', 'date_diff'])

            for pay_idx, group in good_partials.groupby('payment_id_pay_index'):
                pay_rem = pay_df.at[pay_idx, 'amount_remaining']
                
                if pay_rem < TOLERANCE:
                    continue
                    
                for _, row in group.iterrows():
                    inv_idx = row['invoice_id_inv_index']
                    inv_rem = inv_df.at[inv_idx, 'amount_remaining']

                    if inv_rem < TOLERANCE:
                        continue
                    
                    match_amount = min(pay_rem, inv_rem)

                    pay_df.at[pay_idx, 'amount_remaining'] -= match_amount
                    inv_df.at[inv_idx, 'amount_remaining'] -= match_amount
                    
                    pay_rem -= match_amount # Update local copy

                    matches_list.append(Match(
                        payment_id=row['payment_id'],
                        invoice_id=row['invoice_id'],
                        confidence=0.75,
                        rationale=f"Partial/Overpayment match on name/currency",
                        amount_matched=round(match_amount, 2)
                    ))

                    if pay_rem < TOLERANCE:
                        break # Payment is fully consumed

    # --- 5. Final Output ---
    if matches_list:
        matches_df = pd.DataFrame([m.__dict__ for m in matches_list]).drop(columns=['amount_matched'])
    else:
        matches_df = pd.DataFrame(columns=["payment_id", "invoice_id", "confidence", "rationale"])

    final_unmatched_pay = pay_df[pay_df['amount_remaining'] > TOLERANCE].drop(columns=['amount', 'date', 'norm_name', 'amount_remaining'], errors='ignore')
    final_unmatched_inv = inv_df[inv_df['amount_remaining'] > TOLERANCE].drop(columns=['amount', 'date', 'norm_name', 'amount_remaining'], errors='ignore')

    return matches_df, final_unmatched_pay, final_unmatched_inv


def main():
    parser = argparse.ArgumentParser(description="Invoice ↔ Payment matching (starter)")
    parser.add_argument("--invoices", required=True, help="path to invoices.csv")
    parser.add_argument("--payments", required=True, help="path to payments.csv")
    parser.add_argument("--out", default="out/", help="output directory (default: out/)")
    args = parser.parse_args()

    invoices = load_csv(args.invoices).reset_index().rename(columns={'index': 'invoice_id_inv_index'})
    payments = load_csv(args.payments).reset_index().rename(columns={'index': 'payment_id_pay_index'})

    matches, u_pay, u_inv = match_records(invoices, payments)

    os.makedirs(args.out, exist_ok=True)
    
    if 'invoice_id_inv_index' in u_inv.columns:
        u_inv = u_inv.drop(columns=['invoice_id_inv_index'])
    if 'payment_id_pay_index' in u_pay.columns:
        u_pay = u_pay.drop(columns=['payment_id_pay_index'])

    print(f"Writing {len(matches)} matches...")
    write_out(matches, os.path.join(args.out, "matches.csv"))
    
    print(f"Writing {len(u_pay)} unmatched payments...")
    write_out(u_pay, os.path.join(args.out, "unmatched_payments.csv"))
    
    print(f"Writing {len(u_inv)} unmatched invoices...")
    write_out(u_inv, os.path.join(args.out, "unmatched_invoices.csv"))

    summary = {
        "matches": len(matches),
        "unmatched_payments": len(u_pay),
        "unmatched_invoices": len(u_inv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
