
From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

#[export] Instance dec_type (t1 t2 : Typ) : Dec (t1 = t2).
Proof. dec_eq. Defined.

Fixpoint genVar' (ctx : Ctx) (t : Typ) (p : nat) (r : list nat) : list nat :=
  match ctx with
  | nil  => r
  | cons t1 ctx1 => 
    if t = t1? then
      (genVar' ctx1 t (p + 1) (cons p r))
    else
      (genVar' ctx1 t (p + 1) r)
  end.

Fixpoint genZero (env : Ctx) (tau : Typ) : G (option Expr) :=
  match tau with
  | TBool  => 
    (bindGen 
    arbitrary 
    (fun b => 
      (returnGen (Some (Bool b)))))
  | TFun T1 T2 => 
    (bindGen (genZero (cons T1 env) T2) 
    (fun _x => match _x with
      | None  => 
        (returnGen (None ))
      | Some e => 
        (returnGen (Some (Abs T1 e)))
      end))
  end.

Fixpoint genTyp (size : nat) : G (Typ) :=
  match size with
  | O  => 
    (returnGen (TBool ))
  | S size1 => 
    (freq [
      (1, 
      (returnGen (TBool ))); 
      (1, 
      (bindGen (genTyp size1) 
      (fun T1 => 
        (bindGen (genTyp size1) 
        (fun T2 => 
          (returnGen (TFun T1 T2)))))))])
  end.

Fixpoint genExpr (env : list Typ) (tau : Typ) (size : nat) : G (option Expr) :=
  match size with
  | O  => 
    (bindGen 
    (backtrack [
      (1, 
      (oneOf_ 
      (returnGen (None )) 
      (map 
      (fun x => 
        (returnGen (Some (Var x)))) (genVar' env tau 0 (nil ))))); 
      (1, (genZero env tau))]) 
    (fun res => 
      (returnGen res)))
  | S size1 => 
    (bindGen 
    (backtrack [
      (1, 
      (oneOf_ 
      (returnGen (None )) 
      (map 
      (fun x => 
        (returnGen (Some (Var x)))) (genVar' env tau 0 (nil ))))); 
      (1, 
      (bindGen (genTyp 2) 
      (fun T1 => 
        (bindGen (genExpr env (TFun T1 tau) size1) 
        (fun _x => match _x with
          | None  => 
            (returnGen (None ))
          | Some e1 => 
            (bindGen (genExpr env T1 size1) 
            (fun _x => match _x with
              | None  => 
                (returnGen (None ))
              | Some e2 => 
                (returnGen (Some (App e1 e2)))
              end))
          end))))); 
      (1, match tau with
      | TBool  => 
        (bindGen 
        arbitrary 
        (fun b => 
          (returnGen (Some (Bool b)))))
      | TFun T1 T2 => 
        (bindGen (genExpr (cons T1 env) T2 size1) 
        (fun _x => match _x with
          | None  => 
            (returnGen (None ))
          | Some e => 
            (returnGen (Some (Abs T1 e)))
          end))
      end)]) 
    (fun res => 
      (returnGen res)))
  end.

Definition gSized :=

  (bindGen (genTyp 2) 
  (fun typ => (genExpr (nil ) typ 5))).

Definition test_prop_SinglePreserve :=
forAllMaybe gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAllMaybe gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
