From QuickChick Require Import QuickChick. Import QcNotation.

From Coq Require Import Bool ZArith List. Import ListNotations.

From ExtLib Require Import Monad.

From ExtLib.Data.Monads Require Import OptionMonad.

Import MonadNotation.



From STLC Require Import Impl Spec.



Derive (Arbitrary) for Typ.

Derive (Arbitrary) for Expr.



Definition gSized := arbitrary.



Definition test_prop_SinglePreserve :=

  forAll arbitrary (fun (e: Expr) =>

    prop_SinglePreserve e).





(*! QuickChick test_prop_SinglePreserve. *)



Definition test_prop_MultiPreserve :=

  forAll arbitrary (fun (e: Expr) =>

    prop_MultiPreserve e).



(*! QuickChick test_prop_MultiPreserve. *)
            Fixpoint num_apps (e: Expr) : nat :=
                match e with
                | (Abs _ e) => num_apps e
                | (App e1 e2) => 1 + num_apps e1 + num_apps e2
                | _ => 0
                end.
    Definition collect {A : Type} `{_ : Show A} (f : Expr  -> A)  : Checker :=  
        forAll gSized (fun (t : Expr) =>
            if isJust (mt t) then
                collect (append "valid " (show (f t))) true
            else
                collect (append "invalid " (show (f t))) true
        ).
        
Extract Constant Test.defNumTests => "10000".
QuickChick (collect sizeSTLC).
QuickChick (collect num_apps).